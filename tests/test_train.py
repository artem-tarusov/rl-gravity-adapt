import csv
import json
from contextlib import closing
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from gravity_adapt.config import ModelConfig, load_config
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer
from gravity_adapt.ppo import update_ppo
from gravity_adapt.train import collect_rollout, run_training


class BoundaryEnv(gym.Env):
    def __init__(self, terminate: bool):
        self.terminate = terminate
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-0.4, 0.4, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([-10.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        observation = np.array([self.steps + action[0]], dtype=np.float32)
        return observation, float(action[0]), self.terminate, self.steps == 2, {}


def test_rollout_preserves_actions_final_states_and_continuity():
    torch.manual_seed(42)
    model = ActorCritic(1, 1, ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8]))
    with torch.no_grad():
        model.actor.model[-1].weight.zero_()
        model.actor.model[-1].bias.fill_(2.0)
        model.actor.log_std.fill_(-2.0)
    original_weights = {name: value.clone() for name, value in model.state_dict().items()}
    with closing(
        SyncVectorEnv(
            [
                lambda: gym.wrappers.ClipAction(
                    gym.wrappers.RecordEpisodeStatistics(BoundaryEnv(terminate=True))
                ),
                lambda: gym.wrappers.ClipAction(
                    gym.wrappers.RecordEpisodeStatistics(BoundaryEnv(terminate=False))
                ),
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    ) as envs:
        normalizer = Normalizer(1, 2, gamma=0.99)
        raw_observations, _ = envs.reset(seed=42)
        observations = torch.as_tensor(normalizer.normalize_observations(raw_observations))
        rollout, next_observations, episodes = collect_rollout(
            envs, model, normalizer, observations, steps=2
        )
        assert [
            (episode["vector_step"], episode["env_index"], episode["length"])
            for episode in episodes
        ] == [
            (1, 0, 1), (2, 0, 1), (2, 1, 2),
        ]
        np.testing.assert_allclose([episode["return"] for episode in episodes], [0.4, 0.4, 0.8])
        assert rollout["observations"].shape == (2, 2, 1)
        assert rollout["terminated"].tolist() == [[True, False], [True, False]]
        assert rollout["truncated"].tolist() == [[False, False], [False, True]]
        assert (rollout["actions"] > 0.4).all()
        torch.testing.assert_close(rollout["raw_rewards"], torch.full((2, 2), 0.4))
        assert not torch.allclose(rollout["rewards"], rollout["raw_rewards"])
        distribution = model.actor(rollout["observations"])
        torch.testing.assert_close(distribution.log_prob(rollout["actions"]), rollout["log_probs"])
        torch.testing.assert_close(model.critic(rollout["observations"]), rollout["values"])

        expected_final = normalizer.normalize_observations(np.array([[1.4], [2.4]]), update=False)
        torch.testing.assert_close(
            rollout["next_observations"][-1], torch.as_tensor(expected_final)
        )
        assert not torch.allclose(rollout["next_observations"][-1], next_observations)
        assert np.isclose(normalizer.observation_stats.count, 6.0001)
        following, following_observations, _ = collect_rollout(
            envs, model, normalizer, next_observations, steps=1
        )
        torch.testing.assert_close(following["observations"][0], next_observations)
        assert np.isclose(normalizer.observation_stats.count, 8.0001)
        _, _, episodes = collect_rollout(envs, model, normalizer, following_observations, steps=1)
        assert episodes[-1]["length"] == 2
        assert episodes[-1]["return"] == pytest.approx(0.8)
        assert all(
            not tensor.requires_grad and torch.isfinite(tensor).all() for tensor in rollout.values()
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)


def test_training_logs_updates_and_episodes_and_saves_last_rollout(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[1] / "configs" / "earth.yaml"
    config = load_config(path)
    config.environment.num_envs = 2
    config.environment.max_episode_steps = 1
    config.model = ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8])
    config.run.output_dir = str(tmp_path)
    config.ppo.total_steps = 10
    config.ppo.rollout_steps = 3
    config.ppo.minibatch_size = 3
    config.ppo.update_epochs = 2
    updates = []

    def record_update(model, optimizer, rollout, ppo_config):
        before = {name: value.clone() for name, value in model.state_dict().items()}
        metrics = update_ppo(model, optimizer, rollout, ppo_config)
        grad_norm = torch.cat([parameter.grad.flatten() for parameter in model.parameters()]).norm()
        assert grad_norm <= ppo_config.max_grad_norm + 1e-6
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        updates.append((
            before, {int(state["step"].item()) for state in optimizer.state.values()}, rollout
        ))
        return metrics

    monkeypatch.setattr("gravity_adapt.train.update_ppo", record_update)
    run = run_training(config)
    assert load_config(run.directory / "config.yaml") == config
    metadata = json.loads((run.directory / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    metrics = metadata["result"].pop("last_update")
    assert all(np.isfinite(value) for value in metrics.values())
    assert metadata["result"].pop("elapsed_seconds") > 0
    assert metadata["result"] == {
        "kind": "ppo_training",
        "env_steps": 10,
        "rollouts": 2,
        "last_rollout_steps": 2,
        "completed_episodes": 10,
    }
    assert [steps for _, steps, _ in updates] == [{4}, {8}]
    snapshot = torch.load(run.directory / "rollout.pt", weights_only=True)
    rollout = snapshot["rollout"]
    assert rollout["observations"].shape == (2, 2, 348)
    assert rollout["actions"].shape == (2, 2, 17)
    assert rollout["truncated"].all()
    assert rollout["advantages"].shape == rollout["returns"].shape == (2, 2)
    assert all(torch.isfinite(tensor).all() for tensor in rollout.values())
    assert np.isclose(snapshot["normalization"]["observation_count"], 12.0001)
    model = ActorCritic(348, 17, config.model)
    model.load_state_dict(updates[-1][0])
    torch.testing.assert_close(model.critic(rollout["observations"]), rollout["values"])
    torch.testing.assert_close(
        rollout["returns"],
        rollout["rewards"] + config.ppo.gamma * model.critic(rollout["next_observations"]),
    )
    torch.testing.assert_close(
        model.actor(rollout["observations"]).log_prob(rollout["actions"]), rollout["log_probs"]
    )
    model.load_state_dict(snapshot["model"])
    for prefix in ("actor.", "critic."):
        assert any(
            not torch.equal(value, updates[0][0][name])
            for name, value in model.state_dict().items()
            if name.startswith(prefix)
        )
    assert not torch.equal(model.actor.log_std, updates[0][0]["actor.log_std"])
    normalizer = Normalizer(348, 2, config.ppo.gamma)
    normalizer.load_state_dict(snapshot["normalization"])
    assert normalizer.state_dict() == snapshot["normalization"]
    with (run.directory / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    with (run.directory / "episodes.csv").open() as file:
        episodes = list(csv.DictReader(file))
    assert [int(row["env_steps"]) for row in rows] == [6, 10]
    assert [int(row["rollout"]) for row in rows] == [1, 2]
    assert [int(episode["episode"]) for episode in episodes] == list(range(1, 11))
    assert [int(episode["env_steps"]) for episode in episodes] == [
        2, 2, 4, 4, 6, 6, 8, 8, 10, 10,
    ]
    assert all(
        int(episode["length"]) == 1 and episode["truncated"] == "True" for episode in episodes
    )
    output = capsys.readouterr().out
    for row, completed in zip(rows, (episodes[:6], episodes[6:]), strict=True):
        assert "mean_episode_return" not in row
        assert "mean_episode_length" not in row
        mean_return = np.mean([float(episode["return"]) for episode in completed])
        assert f"mean_episode_return={mean_return:.2f}" in output
        assert all(np.isfinite(float(value)) for value in row.values())
