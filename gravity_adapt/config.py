import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
import yaml


@dataclass
class EnvironmentConfig:
    gravity: list[float]
    num_envs: int
    max_episode_steps: int

    def __post_init__(self) -> None:
        if len(self.gravity) != 3:
            raise ValueError("environment.gravity must have three components")


@dataclass
class ModelConfig:
    actor_hidden_sizes: list[int]
    critic_hidden_sizes: list[int]
    lora_rank: int = 0


@dataclass
class RunConfig:
    name: str
    seed: int
    device: Literal["cpu", "mps", "cuda"]
    output_dir: str
    torch_threads: int


@dataclass
class PPOConfig:
    total_steps: int
    rollout_steps: int
    minibatch_size: int
    update_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    critic_loss_weight: float
    max_grad_norm: float


@dataclass
class Config:
    environment: EnvironmentConfig
    model: ModelConfig
    run: RunConfig
    ppo: PPOConfig


def load_config(path: Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = data.keys() - {"environment", "model", "run", "ppo"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(sorted(map(str, unknown)))}")
    return Config(
        environment=EnvironmentConfig(**data["environment"]),
        model=ModelConfig(**data["model"]),
        run=RunConfig(**data["run"]),
        ppo=PPOConfig(**data["ppo"]),
    )


def parse_config(args: list[str] | None = None) -> Config:
    """Load --config first, then apply typed overrides such as --run.seed 43."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    known, remaining = parser.parse_known_args(args)
    return tyro.cli(
        Config,
        args=remaining,
        default=load_config(known.config),
        description="Load all experiment settings with --config PATH, then apply CLI overrides.",
    )
