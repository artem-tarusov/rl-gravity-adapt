from contextlib import closing

import numpy as np
import pytest

from gravity_adapt.config import EnvironmentConfig
from gravity_adapt.envs import make_env, make_vector_env


@pytest.mark.parametrize("gravity", [[0.0, 0.0, -9.81], [0.0, 0.0, -3.71]])
def test_gravity_survives_reset_and_seeds_repeat(gravity):
    config = EnvironmentConfig(gravity=gravity, num_envs=2, max_episode_steps=1000)
    with closing(make_vector_env(config)) as envs:
        first, _ = envs.reset(seed=42)
        envs.step(np.zeros((2, 17)))
        repeated, _ = envs.reset(seed=42)
        np.testing.assert_array_equal(first, repeated)
        assert not np.array_equal(first[0], first[1])
        for env in envs.envs:
            np.testing.assert_array_equal(env.unwrapped.model.opt.gravity, gravity)
        assert envs.single_observation_space.shape == (348,)
        assert envs.single_action_space.shape == (17,)


def test_same_step_reset_preserves_final_observation_and_raw_episode_reward():
    config = EnvironmentConfig(gravity=[0.0, 0.0, -9.81], num_envs=2, max_episode_steps=1)
    with closing(make_vector_env(config)) as envs:
        envs.reset(seed=0)
        observations, rewards, _, truncated, info = envs.step(np.zeros((2, 17)))
        assert truncated.all()
        assert info["_final_obs"].all()
        assert all(np.isfinite(item).all() for item in info["final_obs"])
        assert not np.array_equal(observations[0], info["final_obs"][0])
        np.testing.assert_allclose(info["final_info"]["episode"]["r"], rewards)
        # Every vector step advances a real transition, including the step after reset.
        _, _, _, truncated, _ = envs.step(np.zeros((2, 17)))
        assert truncated.all()


def test_action_clipping_matches_explicitly_bounded_actions():
    config = EnvironmentConfig(gravity=[0.0, 0.0, -9.81], num_envs=1, max_episode_steps=1000)
    with closing(make_env(config)) as env:
        env.reset(seed=1)
        actions = np.full(17, 100.0)
        bounded = np.clip(actions, env.unwrapped.action_space.low, env.unwrapped.action_space.high)
        clipped_observation, clipped_reward, *_ = env.step(actions)
        env.reset(seed=1)
        expected_observation, expected_reward, *_ = env.step(bounded)
        np.testing.assert_array_equal(clipped_observation, expected_observation)
        assert clipped_reward == expected_reward
