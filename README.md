# gravity-adapt

Research into adapting humanoid locomotion policies across gravity regimes.
The stack uses Gymnasium `Humanoid-v5`, MuJoCo, and PyTorch, with separate actor
and critic MLPs. The environment is fixed to `Humanoid-v5` in the code.
The planned study compares training from scratch, full
fine-tuning, and LoRA adaptation.

The current implementation provides configuration, environment creation, actor/critic
networks, run metadata, and a short rollout with an untrained policy. PPO training,
observation/reward normalization for training, checkpoints, and independent policy
evaluation will be added in subsequent stages.

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
    runs.py         # Run directories, metadata, and lifecycle
    smoke.py        # Short rollout with an untrained policy
configs/
    earth.yaml
tests/
    test_config.py
    test_envs.py
    test_models.py
    test_smoke.py
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

## Development checks

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

The tests focus on configuration overrides and essential validation, gravity
persistence, seeded resets, action clipping, final observations, Gaussian log
probabilities, actor/critic gradients, and a short MuJoCo rollout with saved
experiment parameters and results.

Environment behavior follows the [Gymnasium Humanoid documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/).
