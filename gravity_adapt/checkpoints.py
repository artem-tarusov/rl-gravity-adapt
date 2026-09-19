from dataclasses import asdict
from pathlib import Path

import torch

from gravity_adapt.config import Config
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizer: Normalizer,
    config: Config,
    progress: dict,
) -> None:
    rng = {"cpu": torch.get_rng_state()}
    if config.run.device == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state()
    elif config.run.device == "mps":
        rng["mps"] = torch.mps.get_rng_state()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalization": normalizer.state_dict(),
        "config": asdict(config),
        "progress": progress,
        "rng": rng,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    normalizer: Normalizer,
    config: Config,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    current = asdict(config)
    for section in ("environment", "model", "ppo"):
        saved = dict(checkpoint["config"][section])
        requested = dict(current[section])
        if section == "ppo":
            saved.pop("total_steps")
            requested.pop("total_steps")
        if saved != requested:
            raise ValueError(
                f"Resume requires unchanged {section} settings, except ppo.total_steps"
            )
    if config.run.seed != checkpoint["config"]["run"]["seed"]:
        raise ValueError("Resume requires the original run.seed")
    if config.ppo.total_steps <= checkpoint["progress"]["env_steps"]:
        raise ValueError("ppo.total_steps must exceed the checkpoint's env_steps")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    normalizer.load_state_dict(checkpoint["normalization"])
    torch.set_rng_state(checkpoint["rng"]["cpu"])
    if config.run.device == "cuda" and "cuda" in checkpoint["rng"]:
        torch.cuda.set_rng_state(checkpoint["rng"]["cuda"])
    elif config.run.device == "mps" and "mps" in checkpoint["rng"]:
        torch.mps.set_rng_state(checkpoint["rng"]["mps"])
    return checkpoint["progress"]
