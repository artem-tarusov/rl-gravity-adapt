import argparse
import shutil
from contextlib import closing
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch import Tensor

from gravity_adapt.checkpoints import load_checkpoint, save_checkpoint
from gravity_adapt.config import Config, parse_config
from gravity_adapt.envs import make_vector_env
from gravity_adapt.logging import TrainingLogger
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer
from gravity_adapt.ppo import update_ppo
from gravity_adapt.runs import Run, create_run, finish_run


@torch.no_grad()
def collect_rollout(
    envs: SyncVectorEnv,
    model: ActorCritic,
    normalizer: Normalizer,
    observations: Tensor,
    steps: int,
) -> tuple[dict[str, Tensor], Tensor, list[dict]]:
    device = observations.device
    shape = (steps, envs.num_envs)
    observation_shape = (*shape, envs.single_observation_space.shape[0])
    action_shape = (*shape, envs.single_action_space.shape[0])
    rollout = {
        "observations": torch.empty(observation_shape, device=device),
        "next_observations": torch.empty(observation_shape, device=device),
        "actions": torch.empty(action_shape, device=device),
        "log_probs": torch.empty(shape, device=device),
        "values": torch.empty(shape, device=device),
        "rewards": torch.empty(shape, device=device),
        "terminated": torch.empty(shape, dtype=torch.bool, device=device),
        "truncated": torch.empty(shape, dtype=torch.bool, device=device),
    }
    episodes = []
    for step in range(steps):
        actions, log_probs, _, values = model.get_action_and_value(observations)
        rollout["observations"][step] = observations
        rollout["actions"][step] = actions
        rollout["log_probs"][step] = log_probs
        rollout["values"][step] = values

        next_observations, rewards, terminated, truncated, info = envs.step(actions.cpu().numpy())
        done = terminated | truncated
        rollout["rewards"][step] = torch.as_tensor(normalizer.normalize_rewards(rewards, done), device=device)
        rollout["terminated"][step] = torch.as_tensor(terminated, device=device)
        rollout["truncated"][step] = torch.as_tensor(truncated, device=device)
        observations = torch.as_tensor(normalizer.normalize_observations(next_observations), device=device)
        rollout["next_observations"][step] = observations
        if done.any():
            for env_index in np.flatnonzero(done):
                episodes.append({
                    "vector_step": step + 1,
                    "env_index": int(env_index),
                    "return": float(info["final_info"]["episode"]["r"][env_index]),
                    "length": int(info["final_info"]["episode"]["l"][env_index]),
                    "terminated": bool(terminated[env_index]),
                    "truncated": bool(truncated[env_index]),
                })
            # SAME_STEP returns reset observations; PPO needs the preceding episode's final state.
            final_observations = np.stack(info["final_obs"][done])
            rollout["next_observations"][step, torch.as_tensor(done, device=device)] = (
                torch.as_tensor(
                    normalizer.normalize_observations(final_observations, update=False),
                    device=device,
                )
            )
    return rollout, observations, episodes


def run_training(
    config: Config, *, checkpoint: Path | None = None,
    mode: Literal["resume", "finetune"] | None = None,
    checkpoint_interval: int = 1_000_000,
) -> Run:
    if (checkpoint is None) != (mode is None):
        raise ValueError("checkpoint and mode must be provided together")
    if config.ppo.total_steps % config.environment.num_envs:
        raise ValueError("ppo.total_steps must be divisible by environment.num_envs")
    run = create_run(config, checkpoint=checkpoint, mode=mode)
    print(f"Run directory: {run.directory}", flush=True)
    try:
        torch.set_num_threads(config.run.torch_threads)
        torch.manual_seed(config.run.seed)
        device = torch.device(config.run.device)
        with (
            closing(make_vector_env(config.environment)) as envs,
            TrainingLogger(run.directory, config.ppo.total_steps) as logger,
        ):
            normalizer = Normalizer(
                envs.single_observation_space.shape[0], envs.num_envs, config.ppo.gamma
            )
            model = ActorCritic(
                envs.single_observation_space.shape[0],
                envs.single_action_space.shape[0],
                config.model,
            ).to(device)
            optimizer = torch.optim.Adam(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=config.ppo.learning_rate, eps=1e-5,
            )
            progress = {
                "env_steps": 0,
                "rollouts": 0,
                "completed_episodes": 0,
                "elapsed_seconds": 0.0,
            }
            if checkpoint is not None:
                saved_progress = load_checkpoint(checkpoint.expanduser(), model, optimizer, normalizer, config, mode=mode)
                if mode == "resume":
                    progress = saved_progress
                    normalizer.discounted_returns.fill(0.0)
            initial_steps = progress["env_steps"]
            elapsed_before = progress["elapsed_seconds"]
            raw_observations, _ = envs.reset(seed=config.run.seed + initial_steps)
            observations = torch.as_tensor(
                normalizer.normalize_observations(raw_observations), device=device
            )
            total_vector_steps = config.ppo.total_steps // envs.num_envs
            for start in range(
                initial_steps // envs.num_envs, total_vector_steps, config.ppo.rollout_steps
            ):
                learning_rate = config.ppo.learning_rate * (1 - start / total_vector_steps)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                steps = min(config.ppo.rollout_steps, total_vector_steps - start)
                started = run.elapsed_seconds()
                rollout, observations, episodes = collect_rollout(envs, model, normalizer, observations, steps)
                rollout_seconds = run.elapsed_seconds() - started
                started = run.elapsed_seconds()
                metrics = update_ppo(model, optimizer, rollout, config.ppo)
                update_seconds = run.elapsed_seconds() - started
                progress["completed_episodes"] += len(episodes)
                env_steps = (start + steps) * envs.num_envs
                progress.update(
                    env_steps=env_steps,
                    rollouts=progress["rollouts"] + 1,
                    elapsed_seconds=elapsed_before + run.elapsed_seconds(),
                )
                logger.log_episodes(
                    episodes,
                    completed_episodes=progress["completed_episodes"],
                    vector_step_start=start,
                    num_envs=envs.num_envs,
                )
                logger.log_update(
                    progress, metrics, episodes,
                    rollout_seconds=rollout_seconds,
                    update_seconds=update_seconds,
                )
                save_checkpoint(run.directory / "checkpoint.pt", model, optimizer, normalizer, config, progress)
                previous_steps = start * envs.num_envs
                if checkpoint_interval > 0 and (
                    env_steps // checkpoint_interval > previous_steps // checkpoint_interval
                ):
                    directory = run.directory / "checkpoints"
                    directory.mkdir(exist_ok=True)
                    shutil.copyfile(run.directory / "checkpoint.pt", directory / f"{env_steps}.pt")
        finish_run(
            run,
            status="completed",
            result={
                "kind": "ppo_training",
                **progress,
                "last_rollout_steps": steps,
                "last_update": metrics,
            },
        )
    except KeyboardInterrupt:
        finish_run(run, status="interrupted", error="KeyboardInterrupt")
        raise
    except Exception as error:
        finish_run(run, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return run


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, help="Source checkpoint for resume or finetune")
    parser.add_argument("--mode", choices=["resume", "finetune"], help="How to use the checkpoint")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1_000_000,
        help="Transitions between retained checkpoints; 0 keeps only the latest checkpoint",
    )
    known, remaining = parser.parse_known_args()
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    run_training(
        parse_config(remaining), checkpoint=known.checkpoint, mode=known.mode,
        checkpoint_interval=known.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
