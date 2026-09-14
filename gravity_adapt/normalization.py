import numpy as np
from gymnasium.wrappers.utils import RunningMeanStd


class Normalizer:
    def __init__(self, observation_size: int, num_envs: int, gamma: float):
        self.observation_stats = RunningMeanStd(shape=(observation_size,))
        self.return_stats = RunningMeanStd()
        self.discounted_returns = np.zeros(num_envs)
        self.gamma = gamma

    def normalize_observations(
        self, observations: np.ndarray, *, update: bool = True
    ) -> np.ndarray:
        if update:
            self.observation_stats.update(observations)

        normalized = (observations - self.observation_stats.mean) / np.sqrt(self.observation_stats.var + 1e-8)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def normalize_rewards(
        self, rewards: np.ndarray, done: np.ndarray, *, update: bool = True
    ) -> np.ndarray:
        if update:
            self.discounted_returns = self.gamma * self.discounted_returns + rewards
            self.return_stats.update(self.discounted_returns)
            self.discounted_returns[done] = 0.0
        normalized = rewards / np.sqrt(self.return_stats.var + 1e-8)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "observation_mean": self.observation_stats.mean.tolist(),
            "observation_var": self.observation_stats.var.tolist(),
            "observation_count": float(self.observation_stats.count),
            "return_mean": float(self.return_stats.mean),
            "return_var": float(self.return_stats.var),
            "return_count": float(self.return_stats.count),
            "discounted_returns": self.discounted_returns.tolist(),
            "gamma": self.gamma,
        }

    def load_state_dict(self, state: dict) -> None:
        self.observation_stats.mean = np.array(state["observation_mean"])
        self.observation_stats.var = np.array(state["observation_var"])
        self.observation_stats.count = state["observation_count"]
        self.return_stats.mean = np.array(state["return_mean"])
        self.return_stats.var = np.array(state["return_var"])
        self.return_stats.count = state["return_count"]
        self.discounted_returns = np.array(state["discounted_returns"])
        self.gamma = state["gamma"]
