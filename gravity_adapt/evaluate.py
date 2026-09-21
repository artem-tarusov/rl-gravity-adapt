import argparse
import csv
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from gravity_adapt.config import Config, EnvironmentConfig, ModelConfig, PPOConfig, RunConfig
from gravity_adapt.envs import make_env
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer
from gravity_adapt.runs import Run, create_run, finish_run


@torch.no_grad()
def run_evaluation(
    checkpoint: Path,
    *,
    seed: int,
    environment: EnvironmentConfig | None = None,
    episodes: int = 10,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
    output_dir: Path = Path("runs"),
) -> Run:
    checkpoint = checkpoint.expanduser().resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = Config(
        environment=EnvironmentConfig(**saved["config"]["environment"]),
        model=ModelConfig(**saved["config"]["model"]),
        run=RunConfig(**saved["config"]["run"]),
        ppo=PPOConfig(**saved["config"]["ppo"]),
    )
    config.environment = replace(environment or config.environment, num_envs=1)
    config.run.name += "_evaluation"
    config.run.seed = seed
    config.run.device = device
    config.run.torch_threads = 1
    config.run.output_dir = str(output_dir)
    run = create_run(config)
    print(f"Run directory: {run.directory}", flush=True)
    try:
        metadata_path = run.directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["evaluation"] = {
            "checkpoint": str(checkpoint),
            "checkpoint_env_steps": saved["progress"]["env_steps"],
            "episodes": episodes,
            "explore": False,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.set_num_threads(config.run.torch_threads)
        with (
            closing(make_env(config.environment)) as env,
            (run.directory / "episodes.csv").open("w", newline="", encoding="utf-8") as file,
        ):
            model = ActorCritic(
                env.observation_space.shape[0], env.action_space.shape[0], config.model
            ).to(device).eval()
            model.load_state_dict(saved["model"])
            normalizer = Normalizer(env.observation_space.shape[0], 1, config.ppo.gamma)
            normalizer.load_state_dict(saved["normalization"])
            writer = csv.DictWriter(file, fieldnames=[
                "episode", "return", "length", "terminated", "truncated", "x_displacement",
            ])
            writer.writeheader()
            returns = []
            lengths = []
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed) if episode == 0 else env.reset()
                initial_x = float(env.unwrapped.data.qpos[0])
                terminated = truncated = False
                while not (terminated or truncated):
                    observation = torch.as_tensor(
                        normalizer.normalize_observations(observation, update=False), device=device
                    )
                    action, _, _, _ = model.get_action_and_value(observation, explore=False)
                    observation, _, terminated, truncated, info = env.step(action.cpu().numpy())
                episode_return = float(info["episode"]["r"])
                episode_length = int(info["episode"]["l"])
                writer.writerow({
                    "episode": episode + 1,
                    "return": episode_return,
                    "length": episode_length,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "x_displacement": float(env.unwrapped.data.qpos[0]) - initial_x,
                })
                file.flush()
                returns.append(episode_return)
                lengths.append(episode_length)
        finish_run(
            run,
            status="completed",
            result={"kind": "policy_evaluation", "completed_episodes": len(returns)},
        )
    except KeyboardInterrupt:
        finish_run(run, status="interrupted", error="KeyboardInterrupt")
        raise
    except Exception as error:
        finish_run(run, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(
        f"Evaluated {len(returns)} episodes | mean_return={np.mean(returns):.2f} | "
        f"mean_length={np.mean(lengths):.2f}",
        flush=True,
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved policy in fresh environments")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Seed for the first environment reset")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run_evaluation(
        args.checkpoint, seed=args.seed, episodes=args.episodes,
        device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
