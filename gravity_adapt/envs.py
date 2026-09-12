from functools import partial

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from gravity_adapt.config import EnvironmentConfig


def make_env(config: EnvironmentConfig, *, render_mode: str | None = None) -> gym.Env:
    env = gym.make(
        id="Humanoid-v5",
        render_mode=render_mode,
        max_episode_steps=config.max_episode_steps
    )
    env.unwrapped.model.opt.gravity[:] = config.gravity
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return gym.wrappers.ClipAction(env)


def make_vector_env(config: EnvironmentConfig) -> SyncVectorEnv:
    """Use explicit same-step resets; final observations are in info['final_obs']."""
    return SyncVectorEnv(
        [partial(make_env, config) for _ in range(config.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
