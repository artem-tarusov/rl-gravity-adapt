import numpy as np

from gravity_adapt.normalization import Normalizer


def test_observation_normalization_freezes_and_restores():
    normalizer = Normalizer(2, 2, gamma=0.99)
    observations = np.array([[1.0, 4.0], [3.0, 8.0]])
    normalized = normalizer.normalize_observations(observations)
    np.testing.assert_allclose(normalized, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-3)

    state = normalizer.state_dict()
    extreme = np.array([[1e6, -1e6]])
    np.testing.assert_array_equal(
        normalizer.normalize_observations(extreme, update=False), [[10.0, -10.0]]
    )
    assert normalizer.state_dict() == state
    restored = Normalizer(2, 2, gamma=0.5)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    np.testing.assert_array_equal(
        restored.normalize_observations(observations, update=False),
        normalizer.normalize_observations(observations, update=False),
    )


def test_reward_scaling_resets_returns_at_both_episode_boundaries():
    normalizer = Normalizer(1, 2, gamma=0.5)
    normalizer.normalize_rewards(np.array([1.0, 3.0]), np.array([True, False]))
    np.testing.assert_array_equal(normalizer.discounted_returns, [0.0, 3.0])
    normalized = normalizer.normalize_rewards(np.array([2.0, 1.0]), np.array([False, True]))
    expected = np.array([2.0, 1.0]) / np.std([1.0, 3.0, 2.0, 2.5])
    np.testing.assert_allclose(normalized, expected, rtol=1e-3)
    np.testing.assert_array_equal(normalizer.discounted_returns, [2.0, 0.0])

    state = normalizer.state_dict()
    restored = Normalizer(1, 2, gamma=0.99)
    restored.load_state_dict(state)
    rewards, done = np.array([5.0, 7.0]), np.array([False, False])
    restored.normalize_rewards(rewards, done, update=False)
    assert restored.state_dict() == state
    np.testing.assert_array_equal(
        restored.normalize_rewards(rewards, done), normalizer.normalize_rewards(rewards, done)
    )
