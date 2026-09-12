from contextlib import closing
from time import monotonic

import numpy as np
import torch

from gravity_adapt.config import Config, parse_config
from gravity_adapt.envs import make_vector_env
from gravity_adapt.models import ActorCritic
from gravity_adapt.runs import Run, create_run, finish_run


def run_smoke(config: Config, steps: int = 128) -> Run:
    run = create_run(config)
    print(f"Run directory: {run.directory}", flush=True)
    try:
        torch.set_num_threads(config.run.torch_threads)
        torch.manual_seed(config.run.seed)
        device = torch.device(config.run.device)
        with closing(make_vector_env(config.environment)) as envs:
            observations, _ = envs.reset(seed=config.run.seed)
            observation_size = envs.single_observation_space.shape[0]
            action_size = envs.single_action_space.shape[0]
            model = ActorCritic(observation_size, action_size, config.model).to(device).eval()
            completed_returns = []
            completed_lengths = []
            started = monotonic()
            with torch.no_grad():
                for _ in range(steps):
                    batch = torch.as_tensor(observations, dtype=torch.float32, device=device)
                    actions, log_prob, entropy, values = model.get_action_and_value(batch)
                    if not all(
                        torch.isfinite(item).all().item()
                        for item in (actions, log_prob, entropy, values)
                    ):
                        raise RuntimeError("The policy produced non-finite outputs")
                    observations, rewards, terminated, truncated, info = envs.step(
                        actions.cpu().numpy()
                    )
                    if not np.isfinite(observations).all() or not np.isfinite(rewards).all():
                        raise RuntimeError(
                            "The environment produced non-finite observations or rewards"
                        )
                    if "final_info" in info:
                        episode = info["final_info"]["episode"]
                        done = terminated | truncated
                        completed_returns.extend(episode["r"][done].tolist())
                        completed_lengths.extend(episode["l"][done].tolist())
            elapsed = monotonic() - started
            environment_steps = steps * envs.num_envs
            result = {
                "kind": "untrained_policy_smoke",
                "observation_size": observation_size,
                "action_size": action_size,
                "vector_steps": steps,
                "environment_steps": environment_steps,
                "completed_episodes": len(completed_returns),
                "episode_returns": completed_returns,
                "episode_lengths": completed_lengths,
                "rollout_seconds": elapsed,
                "steps_per_second": environment_steps / elapsed,
                "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
                "critic_parameters": sum(p.numel() for p in model.critic.parameters()),
                "autoreset_mode": envs.autoreset_mode.value,
            }
        finish_run(run, status="completed", result=result)
    except KeyboardInterrupt:
        finish_run(run, status="interrupted", error="KeyboardInterrupt")
        raise
    except Exception as error:
        finish_run(run, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(
        f"Smoke check passed: {environment_steps} environment steps; "
        f"completed episodes: {len(completed_returns)}."
    )
    return run


def main() -> None:
    run_smoke(parse_config())


if __name__ == "__main__":
    main()
