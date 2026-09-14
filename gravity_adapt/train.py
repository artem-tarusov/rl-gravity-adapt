from contextlib import closing

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch import Tensor

from gravity_adapt.config import Config, parse_config
from gravity_adapt.envs import make_vector_env
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer
from gravity_adapt.runs import Run, create_run, finish_run


@torch.no_grad()
def collect_rollout(
    envs: SyncVectorEnv,
    model: ActorCritic,
    normalizer: Normalizer,
    observations: Tensor,
    steps: int,
) -> tuple[dict[str, Tensor], Tensor]:
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
        "raw_rewards": torch.empty(shape, device=device),
        "terminated": torch.empty(shape, dtype=torch.bool, device=device),
        "truncated": torch.empty(shape, dtype=torch.bool, device=device),
    }
    for step in range(steps):
        actions, log_probs, _, values = model.get_action_and_value(observations)
        rollout["observations"][step] = observations
        rollout["actions"][step] = actions
        rollout["log_probs"][step] = log_probs
        rollout["values"][step] = values

        next_observations, rewards, terminated, truncated, info = envs.step(actions.cpu().numpy())
        done = terminated | truncated
        rollout["raw_rewards"][step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        rollout["rewards"][step] = torch.as_tensor(normalizer.normalize_rewards(rewards, done), device=device)
        rollout["terminated"][step] = torch.as_tensor(terminated, device=device)
        rollout["truncated"][step] = torch.as_tensor(truncated, device=device)
        observations = torch.as_tensor(normalizer.normalize_observations(next_observations), device=device)
        rollout["next_observations"][step] = observations
        if done.any():
            # SAME_STEP returns reset observations; PPO needs the preceding episode's final state.
            final_observations = np.stack(info["final_obs"][done])
            rollout["next_observations"][step, torch.as_tensor(done, device=device)] = (
                torch.as_tensor(
                    normalizer.normalize_observations(final_observations, update=False),
                    device=device,
                )
            )
    return rollout, observations


def run_training(config: Config) -> Run:
    if config.ppo.total_steps % config.environment.num_envs:
        raise ValueError("ppo.total_steps must be divisible by environment.num_envs")
    run = create_run(config)
    print(f"Run directory: {run.directory}", flush=True)
    print("Collecting experience only; PPO updates are not implemented yet.", flush=True)
    try:
        torch.set_num_threads(config.run.torch_threads)
        torch.manual_seed(config.run.seed)
        device = torch.device(config.run.device)
        with closing(make_vector_env(config.environment)) as envs:
            raw_observations, _ = envs.reset(seed=config.run.seed)
            normalizer = Normalizer(
                envs.single_observation_space.shape[0], envs.num_envs, config.ppo.gamma
            )
            observations = torch.as_tensor(
                normalizer.normalize_observations(raw_observations), device=device
            )
            model = ActorCritic(
                envs.single_observation_space.shape[0],
                envs.single_action_space.shape[0],
                config.model,
            ).to(device)
            total_vector_steps = config.ppo.total_steps // envs.num_envs
            completed_episodes = 0
            rollouts = 0
            for start in range(0, total_vector_steps, config.ppo.rollout_steps):
                steps = min(config.ppo.rollout_steps, total_vector_steps - start)
                rollout, observations = collect_rollout(envs, model, normalizer, observations, steps)
                completed_episodes += int(
                    (rollout["terminated"] | rollout["truncated"]).sum().item()
                )
                rollouts += 1
                environment_steps = (start + steps) * envs.num_envs
                print(
                    f"Collected {environment_steps}/{config.ppo.total_steps} transitions",
                    flush=True,
                )
            torch.save(
                {
                    "rollout": {name: tensor.cpu() for name, tensor in rollout.items()},
                    "model": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                    "normalization": normalizer.state_dict(),
                },
                run.directory / "rollout.pt",
            )
        finish_run(
            run,
            status="completed",
            result={
                "kind": "experience_collection",
                "environment_steps": environment_steps,
                "rollouts": rollouts,
                "last_rollout_steps": steps,
                "completed_episodes": completed_episodes,
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
    run_training(parse_config())


if __name__ == "__main__":
    main()
