import math

import torch

from gravity_adapt.config import ModelConfig
from gravity_adapt.models import ActorCritic


def test_gaussian_log_probability_matches_manual_joint_density():
    config = ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8])
    model = ActorCritic(3, 2, config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    observations = torch.zeros(2, 3)
    actions = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    returned, log_prob, entropy, values = model.get_action_and_value(observations, actions)
    expected = -0.5 * (actions.square() + math.log(2 * math.pi)).sum(dim=-1)
    torch.testing.assert_close(log_prob, expected)
    torch.testing.assert_close(returned, actions)
    torch.testing.assert_close(entropy, torch.full((2,), 1 + math.log(2 * math.pi)))
    torch.testing.assert_close(values, torch.zeros(2))
    mean_actions, *_ = model.get_action_and_value(observations, explore=False)
    torch.testing.assert_close(mean_actions, torch.zeros_like(actions))


def test_actor_and_critic_receive_separate_finite_gradients():
    config = ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8])
    model = ActorCritic(3, 2, config)
    observations = torch.randn(4, 3)
    actions, log_prob, _, values = model.get_action_and_value(observations)
    assert actions.shape == (4, 2)
    assert values.shape == log_prob.shape == (4,)
    actor_ids = {id(parameter) for parameter in model.actor.parameters()}
    assert actor_ids.isdisjoint(id(parameter) for parameter in model.critic.parameters())
    (-log_prob.mean() + (values - 1).square().mean()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
