import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np


class TrainingLogger:
    def __init__(self, directory: Path, total_steps: int):
        self.directory = directory
        self.total_steps = total_steps

    def __enter__(self):
        with ExitStack() as files:
            self.metrics_file = files.enter_context(
                (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8")
            )
            self.episodes_file = files.enter_context(
                (self.directory / "episodes.csv").open("w", newline="", encoding="utf-8")
            )
            self.episode_writer = csv.DictWriter(self.episodes_file, fieldnames=[
                "episode", "env_steps", "env_index", "return", "length",
                "terminated", "truncated",
            ])
            self.episode_writer.writeheader()
            self.metrics_writer = None
            self.files = files.pop_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.files.close()

    def log_episodes(self,episodes: list[dict],*,completed_episodes: int,vector_step_start: int,num_envs: int,) -> None:
        for number, episode in enumerate(episodes, start=completed_episodes - len(episodes) + 1):
            row = dict(episode)
            row["episode"] = number
            row["env_steps"] = (vector_step_start + row.pop("vector_step")) * num_envs
            self.episode_writer.writerow(row)
        self.episodes_file.flush()

    def log_update(
        self,
        progress: dict,
        metrics: dict[str, float],
        episodes: list[dict],
        *,
        rollout_seconds: float,
        update_seconds: float,
    ) -> None:
        row = {
            "rollout": progress["rollouts"],
            "env_steps": progress["env_steps"],
            "elapsed_seconds": progress["elapsed_seconds"],
            "rollout_seconds": rollout_seconds,
            "update_seconds": update_seconds,
            "completed_episodes": progress["completed_episodes"],
            **metrics,
        }
        if self.metrics_writer is None:
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=list(row))
            self.metrics_writer.writeheader()
        self.metrics_writer.writerow(row)
        self.metrics_file.flush()
        mean_return = float(np.mean([episode["return"] for episode in episodes])) if episodes else None
        reward = f"{mean_return:.2f}" if mean_return is not None else "n/a"
        print(
            f"Trained on {progress['env_steps']}/{self.total_steps} transitions | "
            f"mean_episode_return={reward} | "
            f"policy_loss={metrics['policy_loss']:.4f} | "
            f"critic_loss={metrics['critic_loss']:.4f} | "
            f"entropy={metrics['entropy']:.4f}",
            flush=True,
        )
