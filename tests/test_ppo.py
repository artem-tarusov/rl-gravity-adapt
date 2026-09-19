import math
from pathlib import Path

import pytest
import torch

from gravity_adapt.config import ModelConfig, load_config
from gravity_adapt.models import ActorCritic
from gravity_adapt.ppo import compute_gae, update_ppo


def test_gae_handles_termination_truncation_and_rollout_boundary():
    rollout = {
        "rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "values": torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], requires_grad=True),
        "terminated": torch.tensor([[False, False], [True, False], [False, True]]),
        "truncated": torch.tensor([[False, False], [False, True], [False, True]]),
    }
    next_values = torch.tensor([[2.0, 20.0], [100.0, 40.0], [4.0, 999.0]], requires_grad=True)
    advantages, returns = compute_gae(rollout, next_values, gamma=0.9, gae_lambda=0.8)
    torch.testing.assert_close(advantages, torch.tensor([[2.52, 24.4], [1.0, 20.0], [5.6, -24.0]]))
    torch.testing.assert_close(returns, torch.tensor([[3.52, 34.4], [3.0, 40.0], [8.6, 6.0]]))
    assert not advantages.requires_grad
    assert not returns.requires_grad


def test_ppo_clipping_preserves_correct_gradient_direction(monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "earth.yaml").ppo
    config.update_epochs = 1
    config.minibatch_size = 4
    model = ActorCritic(1, 1, ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8]))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    observations = torch.zeros(1, 4, 1)
    actions = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    ratios = torch.tensor([[1.5, 0.5, 1.5, 0.5]])
    rollout = {
        "observations": observations,
        "next_observations": observations.clone(),
        "actions": actions,
        "log_probs": model.actor(observations).log_prob(actions).detach() - ratios.log(),
        "values": torch.zeros(1, 4),
        "rewards": torch.tensor([[1.0, 1.0, -1.0, -1.0]]),
        "terminated": torch.ones(1, 4, dtype=torch.bool),
        "truncated": torch.zeros(1, 4, dtype=torch.bool),
    }
    captured = {}
    get_action_and_value = model.get_action_and_value

    def capture_log_probs(observations, actions):
        result = get_action_and_value(observations, actions)
        result[1].retain_grad()
        captured["log_probs"] = result[1]
        captured["indices"] = actions[:, 0].long()
        return result

    monkeypatch.setattr(model, "get_action_and_value", capture_log_probs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    metrics = update_ppo(model, optimizer, rollout, config)
    assert metrics["policy_loss"] == pytest.approx(0.15, abs=1e-6)
    assert metrics["critic_loss"] == pytest.approx(0.5)
    assert metrics["entropy"] == pytest.approx(0.5 * math.log(2 * math.pi * math.e))
    assert metrics["approx_kl"] == pytest.approx(-0.5 * math.log(0.75), abs=1e-6)
    assert metrics["clip_fraction"] == 1.0
    expected = torch.tensor([0.0, -0.125, 0.375, 0.0])
    torch.testing.assert_close(captured["log_probs"].grad, expected[captured["indices"]])


def test_ppo_entropy_and_critic_weights_control_updates():
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "earth.yaml").ppo
    config.update_epochs = 1
    config.minibatch_size = 2
    config.entropy_coef = 0.2
    config.critic_loss_weight = 0.5
    config.max_grad_norm = 100.0
    model = ActorCritic(1, 1, ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8]))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    observations = torch.zeros(1, 2, 1)
    actions = torch.zeros(1, 2, 1)
    rollout = {
        "observations": observations,
        "next_observations": observations.clone(),
        "actions": actions,
        "log_probs": model.actor(observations).log_prob(actions).detach(),
        "values": torch.zeros(1, 2),
        "rewards": torch.ones(1, 2),
        "terminated": torch.ones(1, 2, dtype=torch.bool),
        "truncated": torch.zeros(1, 2, dtype=torch.bool),
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    update_ppo(model, optimizer, rollout, config)
    torch.testing.assert_close(model.actor.log_std, torch.tensor([0.02]))
    torch.testing.assert_close(model.critic.model[-1].bias, torch.tensor([0.05]))


def test_ppo_metrics_weight_partial_minibatches_by_sample_count(monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "earth.yaml").ppo
    config.update_epochs = 2
    config.minibatch_size = 2
    model = ActorCritic(1, 1, ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8]))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    observations = torch.zeros(1, 3, 1)
    actions = torch.zeros(1, 3, 1)
    rollout = {
        "observations": observations,
        "next_observations": observations.clone(),
        "actions": actions,
        "log_probs": (
            model.actor(observations).log_prob(actions).detach()
            - torch.tensor([[1.0, 1.0, 2.0]]).log()
        ),
        "values": torch.zeros(1, 3),
        "rewards": torch.tensor([[0.0, 0.0, 2.0]]),
        "terminated": torch.ones(1, 3, dtype=torch.bool),
        "truncated": torch.zeros(1, 3, dtype=torch.bool),
    }
    monkeypatch.setattr(torch, "randperm", lambda size, device: torch.arange(size, device=device))
    metrics = update_ppo(model, torch.optim.SGD(model.parameters(), lr=0.0), rollout, config)
    assert metrics["policy_loss"] == pytest.approx(-0.8)
    assert metrics["critic_loss"] == pytest.approx(2 / 3)
    assert metrics["clip_fraction"] == pytest.approx(1 / 3)
    assert metrics["approx_kl"] == pytest.approx((1 - math.log(2)) / 3)
    assert metrics["grad_norm"] == pytest.approx(1 / 3)
