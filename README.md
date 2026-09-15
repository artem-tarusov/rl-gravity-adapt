# gravity-adapt

Research into adapting humanoid locomotion policies across gravity regimes.
The stack uses Gymnasium `Humanoid-v5`, MuJoCo, and PyTorch, with separate actor
and critic MLPs. The environment is fixed to `Humanoid-v5` in the code.
The planned study compares training from scratch, full
fine-tuning, and LoRA adaptation.

The current implementation provides configuration, Humanoid environments, actor/critic
networks, run metadata, a smoke check, and PPO training with normalization and GAE.
The training entry point alternates experience collection and actor/critic updates.
Training logs, resumable checkpoints, and independent policy evaluation will be
added in subsequent stages.

## Setup

The dependency snapshot was verified on Python 3.13.15 and macOS arm64 (Apple Silicon).
Run these commands from the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e ".[dev]"
python -m pip check
```

For the existing environment, activate it with `source .venv/bin/activate`.
The editable installation makes the package importable from other directories
and picks up Python source changes without reinstalling. `--no-build-isolation`
uses the build backend already pinned in the dependency snapshot.

- `pyproject.toml` defines direct dependencies, the `dev` extra (pytest and Ruff),
  package installation, and tool settings.
- `requirements.txt` pins the packages needed by the current implementation,
  including transitive dependencies and development tools, for macOS arm64.
  Install the local package separately with the editable-install command above.
- Configuration paths and relative output paths are resolved against the current
  working directory. Use absolute paths when launching from elsewhere.

### macOS editable-install troubleshooting

If installation succeeds but imports or `gravity-adapt-smoke` fail with
`ModuleNotFoundError`, check whether macOS marked the generated `.pth` file as hidden.
This occurred in the current workspace: Python skipped that file on startup.
The following command clears the hidden flag on this project's installation hook:

```bash
chflags nohidden .venv/lib/python3.13/site-packages/__editable__.gravity_adapt-*.pth
```

In this workspace the hidden flag was reapplied, so the local environment also
contains a direct source symlink. This keeps imports editable even when the `.pth`
hook is skipped. To apply the same workaround to a fresh environment, run from
the project root after the editable installation:

```bash
ln -s "$PWD/gravity_adapt" .venv/lib/python3.13/site-packages/gravity_adapt
```

This link is specific to the local environment. Remove it before switching to a
regular, non-editable installation of this package.

## Project structure

```text
gravity_adapt/
    __init__.py
    config.py       # YAML configuration, validation, and CLI overrides
    envs.py         # Humanoid creation, gravity, action clipping, vectorization
    models.py       # Gaussian actor and separate value network
    normalization.py # Shared observation and discounted-return statistics
    ppo.py          # GAE and PPO minibatch updates
    runs.py         # Run directories, metadata, and lifecycle
    smoke.py        # Short rollout with an untrained policy
    train.py        # Training entry point and rollout collection
configs/
    earth.yaml
tests/
    test_config.py
    test_envs.py
    test_models.py
    test_normalization.py
    test_ppo.py
    test_smoke.py
    test_train.py
runs/               # Generated locally; ignored by Git
```

## Smoke check

```bash
python -m gravity_adapt.smoke --config configs/earth.yaml
```

The equivalent console command supports the same configuration and overrides:

```bash
gravity-adapt-smoke --config configs/earth.yaml --run.seed 43
gravity-adapt-smoke --config configs/earth.yaml --help
```

The `--config` argument is required. The YAML file must provide all sections and
fields; configuration classes define their types without fallback values.
Command-line overrides take precedence over the YAML values.

The supplied `configs/earth.yaml` uses Earth gravity, four synchronous environments,
separate `256 x 256` actor/critic networks with `Tanh`, and CPU inference.
`Tanh` is fixed in the model implementation.
The smoke command runs 128 vector steps: with four environments, this produces
512 environment transitions. Python callers can change the count with
`run_smoke(config, steps=...)`. Environments reset on the same step as an episode
boundary; the final observation remains available in `info["final_obs"]`.
Actions are sampled from a Gaussian and clipped by the environment wrapper.
Reported rewards are raw simulator rewards.

Use `--run.device mps` when Metal is available to the process, or `cuda` on a
compatible server. The supplied configuration uses CPU; choose a training device after a
separate throughput comparison.

Each invocation creates a unique directory under `runs/` with:

- `config.yaml`: the complete YAML configuration after CLI overrides.
- `metadata.json`: Python and core package versions, platform, timestamps, status,
  and rollout diagnostics, including completed-episode returns. The seed and device
  are recorded in `config.yaml`.

The smoke command checks finite policy outputs and simulator observations/rewards.
It records handled failures or interruptions in the run metadata. A process killed
without cleanup may leave its status as `running`. A successful smoke check verifies
integration; the policy has not been trained to walk.

## PPO training

```bash
python -m gravity_adapt.train --config configs/earth.yaml
gravity-adapt-train --config configs/earth.yaml --ppo.total-steps 1024 --ppo.rollout-steps 128
```

Smoke and training use the same `Config` and `earth.yaml`, including the required `ppo`
section. The smoke check does not use PPO settings. Both commands accept YAML/CLI
overrides; use `--config configs/earth.yaml --help` to list available options.
Reinstall the editable package after pulling changes that add console commands.

`ppo.total_steps` counts transitions across all environments and must be divisible
by `environment.num_envs`. `ppo.rollout_steps` counts steps per environment in each
rollout. The supplied configuration collects two rollouts of 2,048 vector steps,
for 16,384 transitions in total. A smaller final rollout uses the remaining budget.
After each rollout, GAE estimates advantages and critic targets from normalized
rewards. Terminations have no value bootstrap. Time-limit truncations bootstrap
from the episode's final observation; both boundaries stop the GAE recursion.
An unfinished episode at a rollout boundary bootstraps from its next observation.
This follows Gymnasium's [guidance on time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

PPO shuffles the collected transitions for each of `update_epochs` passes and
updates the model in groups of `minibatch_size`, including a smaller final group.
Advantages are centered and divided by their population standard deviation within
each minibatch, with `1e-8` added to the denominator. A singleton minibatch keeps
its original advantage. Stored action log probabilities and critic targets stay
fixed throughout these passes.

The objective combines the clipped policy loss, `critic_loss_weight` times half
the mean squared critic error, and the negative entropy bonus weighted by
`entropy_coef`. The critic loss is not clipped. Adam uses the configured constant
`learning_rate` and `eps=1e-5`; the combined gradient norm is limited by
`max_grad_norm` before each optimizer step. The default configuration performs
32 minibatch updates per epoch, 10 epochs per rollout, and two rollouts: 640
optimizer steps in total. This is a short implementation check, not a full
Humanoid training budget.

Observations are centered and scaled using statistics shared across the vector
environments. Rewards are divided by the running standard deviation of discounted
returns; the return accumulator resets after either termination or truncation.
Both normalized observations and rewards are clipped to `[-10, 10]`, with `1e-8`
added to variance before taking the square root. Normalization statistics can be
saved, restored, and frozen with `update=False` for evaluation.

The collector keeps tensors with leading dimensions `[rollout_steps, num_envs]`:
observations, next observations, actions, action log probabilities, critic values,
normalized rewards, raw rewards, and separate termination/truncation flags.
Stored observations are the exact normalized inputs used by the policy. Actions
and log probabilities describe the original Gaussian samples, before action clipping.
For completed episodes, stored next observations come from `info["final_obs"]` and
use the current normalization statistics without updating them. The next policy
call receives the reset observation. Environments and normalization state continue
across rollout boundaries.

Each console progress line includes the mean raw episode return for episodes that
ended during that rollout, including rewards collected in earlier rollouts. It
reports `n/a` if no episodes ended. The same line reports policy loss, critic loss,
and entropy from the
last minibatch of each PPO update. Final metadata also includes its approximate
KL divergence and clipping fraction under `result.last_minibatch`. These values
are computed before that minibatch's optimizer step, rather than averaged over
the rollout or evaluated after all updates.

Each training run saves its configuration, metadata, and `rollout.pt`. This
diagnostic snapshot contains the last rollout with its GAE advantages and critic
targets (`returns`), model weights after the final PPO update, and normalization
state. The rollout's actions, log probabilities, and values were collected before
that update, so the saved model generally produces different predictions.
Read the snapshot with `torch.load(path, weights_only=True)`. It is not a resumable
training checkpoint; optimizer and random-number generator states are not saved.
CSV logging and resumable checkpoints are subsequent implementation steps.

PPO settings and normalization conventions use
[CleanRL's continuous-action PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)
as a reference. Running statistics use Gymnasium's `RunningMeanStd` implementation.

## Development checks

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

The tests focus on configuration overrides and essential validation, gravity
persistence, seeded resets, action clipping, final observations, Gaussian log
probabilities, actor/critic gradients, and a short MuJoCo rollout with saved
experiment parameters and results. Collection checks cover final observations,
unclipped action probabilities, normalization freezing/restoration, and continuity
between rollouts. PPO checks cover hand-calculated GAE targets, clipping gradients,
entropy and critic loss weights, partial minibatches, optimizer step counts, and
changes to both actor and critic weights.

Environment behavior follows the [Gymnasium Humanoid documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/).
