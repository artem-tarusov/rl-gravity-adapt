# rl-gravity-adapt

Humanoid locomotion policy adaptation from Earth gravity to Mars gravity.
The project compares training from scratch, full fine-tuning, and LoRA using
PPO, Gymnasium's `Humanoid-v5`, MuJoCo, and PyTorch.

## Installation

Requires Python 3.13 or later. Pinned dependencies were tested with Python 3.13.15
on macOS arm64 (Apple Silicon).

Run from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
```

## Usage

Run commands from the repository root. Settings are stored in [configs/](configs/)
and can be overridden on the command line. The supplied training configurations
use CPU and a budget of 5,000,000 transitions.

Each training run saves its configuration, logs, and `checkpoint.pt` under `runs/`
and prints the directory path. Checkpoints and videos are generated locally and
are not included in the repository.

### Train on Earth

```bash
python -m gravity_adapt.train --config configs/earth.yaml
```

To train from scratch on Mars, use `configs/mars.yaml` instead.

### Fine-tune on Mars

Replace `EARTH_RUN_ID` with the directory name from Earth training.

```bash
python -m gravity_adapt.train --config configs/mars_finetune.yaml \
    --checkpoint runs/EARTH_RUN_ID/checkpoint.pt --mode finetune
```

### Adapt with LoRA

This example applies rank-64 LoRA to the actor and fully fine-tunes the critic.

```bash
python -m gravity_adapt.train --config configs/mars_lora_actor_rank64.yaml \
    --checkpoint runs/EARTH_RUN_ID/checkpoint.pt --mode finetune
```

For LoRA on both networks, use `configs/mars_lora_actor_critic_rank64.yaml`.
Both variants have configurations and results for ranks 4, 16, 32, and 64 below.

### Evaluate a checkpoint

Replace `RUN_ID` with the training or adaptation directory name.
Evaluation uses the checkpoint's gravity setting.

```bash
python -m gravity_adapt.evaluate --checkpoint runs/RUN_ID/checkpoint.pt \
    --seed 10000 --episodes 100
```

Add `--video` to record evaluation episodes.
For interactive training, evaluation, and plots, see the
[Earth pilot notebook](notebooks/earth_pilot.ipynb).

List available command-line options:

```bash
python -m gravity_adapt.train --config configs/earth.yaml --help
python -m gravity_adapt.evaluate --help
```

## Experiments

Each experiment covers ten training seeds and 100 evaluation episodes per model.
Reports include the protocol, per-seed summaries, and per-episode data.

| Experiment | Configuration | Report |
|---|---|---|
| Earth baseline | [earth.yaml](configs/earth.yaml) | [Results](results/earth_baseline/report.md) |
| Mars baseline | [mars.yaml](configs/mars.yaml) | [Results](results/mars_baseline/report.md) |
| Earth → Mars without adaptation | Earth checkpoints evaluated on Mars | [Results](results/earth_to_mars/report.md) |
| Earth → Mars full fine-tuning | [mars_finetune.yaml](configs/mars_finetune.yaml) | [Results](results/earth_to_mars_finetune/report.md) |
| Actor LoRA, rank 4 | [Configuration](configs/mars_lora_actor_rank4.yaml) | [Results](results/earth_to_mars_lora_actor_rank4/report.md) |
| Actor LoRA, rank 16 | [Configuration](configs/mars_lora_actor_rank16.yaml) | [Results](results/earth_to_mars_lora_actor_rank16/report.md) |
| Actor LoRA, rank 32 | [Configuration](configs/mars_lora_actor_rank32.yaml) | [Results](results/earth_to_mars_lora_actor_rank32/report.md) |
| Actor LoRA, rank 64 | [Configuration](configs/mars_lora_actor_rank64.yaml) | [Results](results/earth_to_mars_lora_actor_rank64/report.md) |
| Actor + Critic LoRA, rank 4 | [Configuration](configs/mars_lora_actor_critic_rank4.yaml) | [Results](results/earth_to_mars_lora_actor_critic_rank4/report.md) |
| Actor + Critic LoRA, rank 16 | [Configuration](configs/mars_lora_actor_critic_rank16.yaml) | [Results](results/earth_to_mars_lora_actor_critic_rank16/report.md) |
| Actor + Critic LoRA, rank 32 | [Configuration](configs/mars_lora_actor_critic_rank32.yaml) | [Results](results/earth_to_mars_lora_actor_critic_rank32/report.md) |
| Actor + Critic LoRA, rank 64 | [Configuration](configs/mars_lora_actor_critic_rank64.yaml) | [Results](results/earth_to_mars_lora_actor_critic_rank64/report.md) |

## License and acknowledgments

Released under the [MIT License](LICENSE).
Parts of the PPO implementation are adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl).
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and license terms.
