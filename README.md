# gravity-adapt

Research into adapting humanoid locomotion policies across gravity regimes.
The stack uses Gymnasium `Humanoid-v5`, MuJoCo, and PyTorch, with separate actor
and critic MLPs. The environment is fixed to `Humanoid-v5` in the code.
The planned study compares training from scratch, full
fine-tuning, and LoRA adaptation.

The current implementation provides configuration, Humanoid environments, actor/critic
networks, run metadata, and PPO training with normalization and GAE.
The training entry point alternates experience collection and actor/critic updates.
Training writes CSV logs and checkpoints that can restore model and optimizer
state in a new run. Independent evaluation loads a checkpoint and records raw
episode results with frozen observation normalization.

The fixed `earth_baseline` uses 5,000,000 transitions and five PPO update epochs.
Verification across ten training seeds is recorded in the
[Earth baseline report](results/earth_baseline/report.md).

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

If installation succeeds but imports or `gravity-adapt-train` fail with
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
    checkpoints.py  # Training state, RNG restoration, and checkpoint replacement
    config.py       # YAML configuration, validation, and CLI overrides
    envs.py         # Humanoid creation, gravity, action clipping, vectorization
    evaluate.py     # Independent checkpoint evaluation and episode videos
    logging.py      # Training CSV logs and console progress
    lora.py         # Frozen linear layers with trainable low-rank updates
    models.py       # Gaussian actor and separate value network
    normalization.py # Shared observation and discounted-return statistics
    ppo.py          # GAE and PPO minibatch updates
    runs.py         # Run directories, metadata, and lifecycle
    train.py        # Training entry point and rollout collection
configs/
    earth.yaml
    mars.yaml
    mars_finetune.yaml
    mars_lora_actor_rank4.yaml
    mars_lora_actor_rank16.yaml
    mars_lora_actor_rank32.yaml
    mars_lora_actor_rank64.yaml
    mars_lora_actor_critic_rank64.yaml
notebooks/
    earth_pilot.ipynb
tests/
    test_checkpoints.py
    test_config.py
    test_envs.py
    test_evaluate.py
    test_models.py
    test_normalization.py
    test_ppo.py
    test_train.py
runs/               # Generated locally; ignored by Git
```

## Configuration and runs

Training requires `--config`. The YAML file must provide all sections and
required fields. `model.actor_lora_rank` and `model.critic_lora_rank` default to `0` (disabled).
Command-line overrides take precedence over the YAML values.
Use `--model.actor-lora-rank` to override the actor rank. Saved configurations and
checkpoints using the previous `model.lora_rank` name remain readable.

The supplied `configs/earth.yaml` uses Earth gravity, four synchronous environments,
separate `256 x 256` actor/critic networks with `Tanh`, and CPU training.
`Tanh` is fixed in the model implementation.
Environments reset on the same step as an episode
boundary; the final observation remains available in `info["final_obs"]`.
Actions are sampled from a Gaussian and clipped by the environment wrapper.

Use `--run.device mps` when Metal is available to the process, or `cuda` on a
compatible server. The supplied configuration uses CPU; choose a training device after a
separate throughput comparison.

Each invocation creates a unique directory under `runs/` with:

- `config.yaml`: the complete YAML configuration after CLI overrides.
- `metadata.json`: Python and core package versions, platform, timestamps, status,
  and run results. The seed and device
  are recorded in `config.yaml`.

Handled failures or interruptions are recorded in the run metadata. A process
killed without cleanup may leave its status as `running`.

## PPO training

```bash
python -m gravity_adapt.train --config configs/earth.yaml
gravity-adapt-train --config configs/earth.yaml --ppo.total-steps 1024 --ppo.rollout-steps 128
```

Use a short training run, such as the 1,024-transition command above, to check
integration. Training accepts YAML/CLI overrides; use `--config configs/earth.yaml --help`
to list available options.
Reinstall the editable package after pulling changes that add console commands.

`ppo.total_steps` counts transitions across all environments and must be divisible
by `environment.num_envs`. `ppo.rollout_steps` counts steps per environment in each
rollout. The baseline configuration collects 8,192 transitions per full rollout,
for 5,000,000 transitions in total. A smaller final rollout uses the remaining budget.
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
`entropy_coef`. The critic loss is not clipped. Adam uses `eps=1e-5` and a learning
rate that decreases linearly from `ppo.learning_rate` toward zero. Before each
rollout, it is set to `ppo.learning_rate * (1 - completed_steps / ppo.total_steps)`.
The last update still uses a positive rate. The combined gradient norm is limited by
`max_grad_norm` before each optimizer step. The default configuration performs
32 minibatch updates per epoch and 5 epochs per full rollout.

Observations are centered and scaled using statistics shared across the vector
environments. Rewards are divided by the running standard deviation of discounted
returns; the return accumulator resets after either termination or truncation.
Both normalized observations and rewards are clipped to `[-10, 10]`, with `1e-8`
added to variance before taking the square root. Normalization statistics can be
saved, restored, and frozen with `update=False` for evaluation.

The collector keeps tensors with leading dimensions `[rollout_steps, num_envs]`:
observations, next observations, actions, action log probabilities, critic values,
normalized rewards, and separate termination/truncation flags.
Stored observations are the exact normalized inputs used by the policy. Actions
and log probabilities describe the original Gaussian samples, before action clipping.
For completed episodes, stored next observations come from `info["final_obs"]` and
use the current normalization statistics without updating them. The next policy
call receives the reset observation. Environments and normalization state continue
across rollout boundaries.

Each console progress line includes the mean raw episode return for episodes that
ended during that rollout, including rewards collected in earlier rollouts. It
reports `n/a` if no episodes ended. Policy loss, critic loss, and entropy are now
averaged over all minibatches and epochs in the PPO update, weighted by minibatch
size. Approximate KL, clipping fraction, and the gradient norm before clipping use
the same averaging. These are measurements during optimization, not a separate
evaluation of the final policy. Final metadata stores them under `result.last_update`.

### Logs and checkpoints

Alongside `config.yaml` and `metadata.json`, each training run writes:

- `metrics.csv`: one row per completed PPO update. Columns include the global
  rollout and environment-step counters, cumulative completed episodes, elapsed
  time, collection and update durations, and the six PPO metrics described above.
  Compute throughput during analysis from the increase in `env_steps` divided by
  `rollout_seconds + update_seconds`, using the saved checkpoint counter as the
  starting value for resumed runs. The initial learning rate is in `config.yaml`.
- `episodes.csv`: one row per completed episode, with its global episode number,
  environment-step counter at completion, environment index, raw return, length,
  and termination/truncation flags. Returns include the full episode even when it
  spans multiple rollouts; normalization does not change these reported rewards.
  Compute episode averages from this file during analysis; they are not stored
  in `metrics.csv`. Mean episode return is still shown in the console.
- `checkpoint.pt`: the latest model and Adam states, normalization statistics and
  accumulators, configuration, training counters, cumulative elapsed time, and
  PyTorch CPU and active-device RNG states. Adam's state includes its step counts,
  moments, and the learning rate used for the latest update.

CSV files are flushed after each update. The checkpoint is then written to a
temporary file and replaces the previous checkpoint only after serialization
succeeds. An interruption leaves the previous completed checkpoint available;
the first checkpoint appears after the first completed update. There is one latest
checkpoint at the run root; periodic copies are also retained as described below.
Read checkpoints with `torch.load(path, map_location="cpu", weights_only=True)`.

### Intermediate checkpoints

Training retains a checkpoint every 1,000,000 transitions, after the rollout that
crosses the interval boundary. Files are saved as `checkpoints/ENV_STEPS.pt` inside
the training run, using the actual transition count. Rollouts are not shortened;
if one rollout crosses several boundaries, it produces one retained checkpoint.
The root `checkpoint.pt` continues to update after every rollout and contains the
final state when training finishes.

```bash
python -m gravity_adapt.train --config configs/earth.yaml --checkpoint-interval 1000000
```

Python callers can pass `checkpoint_interval` to `run_training`. This option is a
training argument, not a YAML field. Set it to `0` to keep only the root checkpoint.
Resume retains new checkpoints using the cumulative transition count; existing
files in the source run remain unchanged.

Training does not run evaluation. Evaluate any saved checkpoint separately:

```bash
python -m gravity_adapt.evaluate \
    --checkpoint runs/RUN_ID/checkpoints/1007616.pt --seed 10000 --episodes 100
```

Replace the example path with a retained checkpoint. The pilot notebook's
independent evaluation cell uses the `checkpoint` variable; set it to a saved
intermediate checkpoint to evaluate that version. If evaluations are used to
select a checkpoint, use a different evaluation seed for the final comparison.

### Resume training

Use the saved configuration to preserve the settings of the original run:

```bash
python -m gravity_adapt.train \
    --config runs/RUN_ID/config.yaml \
    --checkpoint runs/RUN_ID/checkpoint.pt --mode resume \
    --ppo.total-steps 32768
```

Replace `RUN_ID` with the source run directory name. `--checkpoint` and `--mode`
are training CLI arguments, not YAML fields. Provide both to load a checkpoint;
omit both to train from scratch. Python callers can use
`run_training(config, checkpoint=path, mode="resume")`.
`ppo.total_steps` remains the total budget including already completed transitions:
resuming at 16,384 with a target of 32,768 collects another 16,384 transitions.
The learning rate is recalculated from the saved transition count and the requested
total budget. Increasing the budget can therefore increase the learning rate on resume.
The target must exceed the saved counter. Environment, model, PPO settings other
than the total budget, and seed must match the checkpoint. Run name, output
directory, device, and thread count may change.

Resuming creates a new run directory and records the source path in
`metadata.json` as `resumed_from`. Existing artifacts are preserved. New CSV files
contain only the continuation, with cumulative counters and elapsed time carried
forward; join them with the source logs to plot the full training history.
Use the checkpoint counter as the restart boundary if an interrupted run has CSV
rows beyond its last successful checkpoint. Idle time between runs is excluded.

Model, optimizer, normalization statistics, and available matching-device PyTorch
RNG states are restored. MuJoCo states and environment RNG states are not restored:
all environments start fresh episodes using `run.seed + saved_env_steps`.
Reward accumulators are cleared and reset observations update the existing
observation statistics. Unfinished episodes from the source run are not logged
as completed episodes. Resume therefore continues optimization without reproducing
the exact uninterrupted trajectory; changing devices can also change random
sequences and numerical results. Evaluation uses frozen observation statistics.

### Fine-tuning

Start a new training run from an existing model with `--mode finetune`:

```bash
python -m gravity_adapt.train \
    --config configs/mars_finetune.yaml \
    --checkpoint runs/EARTH_RUN_ID/checkpoint.pt --mode finetune
```

Python callers can use `run_training(config, checkpoint=path, mode="finetune")`.
Fine-tuning loads the actor, including `log_std`, the critic, and observation and
return normalization statistics. With LoRA disabled, all model parameters remain trainable. Both sets
of normalization statistics continue updating from their saved counts during training.
Discounted-return accumulators start at zero for the new environments, using the
new configuration's `gamma`.

Adam, training counters, elapsed time, and the learning-rate schedule start fresh.
The new run's seed initializes random state; checkpoint RNG state is not restored.
`ppo.total_steps` counts only new transitions. The supplied configuration uses
Mars gravity, 5,000,000 new transitions, five PPO epochs, and learning rate 1e-4
with linear decay. The model architecture must match the source checkpoint.

Metadata records the source checkpoint as `initialized_from`. Logs and checkpoints
use the existing format and periodic saving mechanism. To continue interrupted
fine-tuning, use its saved configuration and checkpoint with `--mode resume`.
Evaluate saved checkpoints separately with the evaluation entry point.

Parts of the PPO implementation are adapted from
[CleanRL's continuous-action PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy).
See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution and license terms. Running statistics use Gymnasium's `RunningMeanStd` implementation.

### LoRA adaptation

Actor LoRA experiments with a fully trainable critic: [rank 4](results/earth_to_mars_lora_actor_rank4/report.md) · [rank 16](results/earth_to_mars_lora_actor_rank16/report.md) · [rank 32](results/earth_to_mars_lora_actor_rank32/report.md) · [rank 64](results/earth_to_mars_lora_actor_rank64/report.md).

Actor and critic LoRA experiment: [rank 64](results/earth_to_mars_lora_actor_critic_rank64/report.md).

Use the same fine-tuning entry point with `model.actor_lora_rank: 4`:

```bash
python -m gravity_adapt.train \
    --config configs/mars_lora_actor_rank4.yaml \
    --checkpoint runs/EARTH_RUN_ID/checkpoint.pt --mode finetune
```

[LoRA](https://arxiv.org/abs/2106.09685) adds a trainable low-rank update to each
actor linear layer, including the output layer. Base weights and biases are
frozen; `log_std` remains trainable. Set `model.critic_lora_rank` to apply LoRA
to each critic linear layer as well; its default of `0` keeps the entire critic trainable.
`configs/mars_lora_actor_critic_rank64.yaml` uses rank 64 for both networks.
The update is `B @ A`,
with scaling fixed at 1 (`alpha = rank`). `A` uses Kaiming uniform initialization
and `B` starts at zero, preserving the source network outputs before training.

LoRA uses the existing PPO settings and normalization updates.
Checkpoints store base weights and adapters
together. Resume and evaluation reconstruct the adapters from the saved ranks
and use the existing commands.

## Independent evaluation

```bash
gravity-adapt-evaluate --checkpoint runs/RUN_ID/checkpoint.pt --seed 10000 --episodes 10
```

Reinstall the editable package to register the command, or use
`python -m gravity_adapt.evaluate` with the same arguments. No separate configuration
file is needed: environment and model settings come from the checkpoint.
Evaluation defaults to CPU and writes under `runs`; override these with `--device`
and `--output-dir`. Evaluation uses one PyTorch CPU thread.

Python callers can pass `environment=EnvironmentConfig(...)` to `run_evaluation`
to evaluate the same policy under different gravity or an episode time limit.
The supplied settings are saved in the evaluation's `config.yaml`; evaluation
still uses one environment, and the caller's configuration is not modified.
Without this argument, environment settings come from the checkpoint. Model weights
and normalization statistics always come from the checkpoint.

Evaluation uses one fresh environment and the Gaussian policy's mean actions
(`explore=False`), with the usual environment action clipping. Observation statistics
are loaded and frozen; rewards are not normalized. Model weights remain unchanged,
and training optimizer and RNG states are not restored.

Choose an evaluation seed separately from the training seed. It is passed only to
the first reset; later episodes use `reset()` to continue the environment's random
sequence. Repeating the same evaluation with the same software and device reproduces
the episode sequence. The seed is saved once in `config.yaml`, not in each CSV row.

Each evaluation creates a separate run directory containing:

- `config.yaml`: checkpoint settings with the actual evaluation seed, device,
  output directory, and one environment. Retained PPO settings describe training
  and are not used to update the policy.
- `metadata.json`: checkpoint path and saved `env_steps`, requested episode count,
  action-selection mode, runtime information, and completion or interruption status.
- `episodes.csv`: episode number, raw return, length, termination/truncation flags,
  and `x_displacement` (final minus initial root-body X position in meters).

CSV rows are flushed after each completed episode; an interrupted episode is not
recorded as complete. Mean return and length are printed in the console. Compute
aggregates from the CSV during analysis. The checkpoint is left unchanged.

### Episode videos

Add `--video` to save every evaluated episode as an MP4:

```bash
gravity-adapt-evaluate --checkpoint runs/RUN_ID/checkpoint.pt --seed 10000 --episodes 3 --video
```

Python callers can pass `video=True` to `run_evaluation`. Gymnasium's `RecordVideo`
captures the initial state and each subsequent step, including the final state.
It uses the environment's frame rate (67 FPS for the current Humanoid settings).
Videos are written under the evaluation run's `videos/` directory; Gymnasium names
episodes starting at zero, so `evaluation-episode-0.mp4` corresponds to episode 1
in `episodes.csv`. All requested episodes are recorded, including falls.

Recording uses the same actions and seeded episode sequence as evaluation without
video. It adds rendering and encoding time. On interruption, a partial video may
be saved; only completed episodes appear in the CSV. Rendering requires a graphics
context, including access to macOS graphics services when running in a sandbox.
MoviePy and its FFmpeg dependency handle MP4 encoding. Videos stay under `runs/`
and are ignored by Git.

## Pilot notebook

Open `notebooks/earth_pilot.ipynb` in a notebook-capable editor. Install the notebook
dependencies and register the project's Python kernel:

```bash
python -m pip install --no-build-isolation -e ".[notebook]"
python -m ipykernel install --prefix .venv --name gravity-adapt --display-name "Python (gravity-adapt)"
```

Select the `Python (gravity-adapt)` kernel or the project's `.venv/bin/python`.
The notebook supports a working directory of either the project root or `notebooks/`.
It calls `run_training` and `run_evaluation`, reads saved CSV data, plots learning
curves, PPO diagnostics, throughput, and evaluation results, and displays recorded video.

The pilot uses five million transitions on CPU. Run the cells in order to start
new training, evaluate its checkpoint, and analyze the results. `record_video`
controls video recording. Rerunning the training or evaluation cell creates a new run.
Restart the kernel after changing the training code so that new imports use it.

Keep notebook outputs cleared before committing. Experiment data and videos remain
in the ignored `runs/` directory, and `.ipynb_checkpoints/` is ignored too.

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
Logging and checkpoint checks cover raw episode statistics, weighted PPO metrics,
restoration of Adam and RNG state, continuation counters, fresh episodes on resume,
and preservation of the previous checkpoint when writing is interrupted.
Evaluation checks cover mean actions, frozen weights and normalization, raw results,
seeded episode sequences, interruption status, and MP4 recording without changes
to episode results. The recording test uses synthetic frames to avoid requiring
a graphics context in the test suite.

Environment behavior follows the [Gymnasium Humanoid documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/).

## Licensing

No project license is currently granted for gravity-adapt's original contributions.
Third-party portions retain their existing licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md).
