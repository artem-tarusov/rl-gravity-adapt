import math

import pytest
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


@pytest.mark.parametrize("network", ["actor", "critic"])
def test_lora_requires_a_positive_rank(network):
    config = ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8])
    setattr(config, f"{network}_adaptation", "lora")
    with pytest.raises(ValueError, match="positive"):
        ActorCritic(3, 2, config)


@pytest.mark.parametrize("adaptation", ["full"])
def test_explicit_adaptation_takes_precedence_over_unused_lora_ranks(adaptation):
    config = ModelConfig(
        actor_hidden_sizes=[8], critic_hidden_sizes=[8],
        actor_adaptation=adaptation, critic_adaptation=adaptation,
        actor_lora_rank=4, critic_lora_rank=4,
    )
    model = ActorCritic(3, 2, config)
    assert not any("lora" in name for name, _ in model.named_parameters())
    for network in (model.actor, model.critic):
        assert network.model[0].weight.requires_grad == (adaptation == "full")
        assert network.model[-1].weight.requires_grad
