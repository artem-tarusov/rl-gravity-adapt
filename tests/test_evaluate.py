import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from imageio.v2 import get_reader

from gravity_adapt.checkpoints import save_checkpoint
from gravity_adapt.config import EnvironmentConfig, ModelConfig, load_config
from gravity_adapt.envs import make_env
from gravity_adapt.evaluate import run_evaluation
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer


@pytest.fixture
def checkpoint(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "earth.yaml")
    config.environment.max_episode_steps = 2
    config.model = ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8])
    torch.set_num_threads(1)
    torch.manual_seed(config.run.seed)
    model = ActorCritic(348, 17, config.model)
    normalizer = Normalizer(348, config.environment.num_envs, config.ppo.gamma)
    normalizer.normalize_observations(np.stack([np.zeros(348), np.full(348, 2.0)]))
    normalizer.normalize_rewards(np.ones(config.environment.num_envs), np.zeros(4, dtype=bool))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, torch.optim.Adam(model.parameters()), normalizer, config,
        {"env_steps": 8, "rollouts": 1, "completed_episodes": 0, "elapsed_seconds": 1.0},
    )
    return path


@pytest.mark.parametrize("environment", [
    None,
    EnvironmentConfig(gravity=[0.0, 0.0, -3.71], num_envs=4, max_episode_steps=3),
])
def test_evaluation_freezes_policy_and_statistics_and_repeats_episode_sequence(
    checkpoint, tmp_path, monkeypatch, environment,
):
    saved = torch.load(checkpoint, weights_only=True)
    expected_environment = environment or EnvironmentConfig(**saved["config"]["environment"])
    original = checkpoint.read_bytes()
    resets = []
    raw_episodes = []
    models = []
    normalizers = []
    original_action = ActorCritic.get_action_and_value
    original_normalize = Normalizer.normalize_observations

    def check_action(self, observation, *, explore):
        assert not self.training and not torch.is_grad_enabled() and not explore
        result = original_action(self, observation, explore=explore)
        torch.testing.assert_close(result[0], self.actor(observation).mean)
        models.append(self)
        return result

    def check_normalize(self, observation, *, update):
        assert not update
        assert self.state_dict() == saved["normalization"]
        result = original_normalize(self, observation, update=update)
        normalizers.append(self)
        return result

    def record_env(config, **kwargs):
        assert config.num_envs == 1
        env = make_env(config, **kwargs)
        np.testing.assert_array_equal(env.unwrapped.model.opt.gravity, expected_environment.gravity)
        reset, step = env.reset, env.step
        total, initial_x = 0.0, 0.0

        def record_reset(**kwargs):
            nonlocal total, initial_x
            observation, info = reset(**kwargs)
            resets.append((kwargs, observation.copy()))
            total, initial_x = 0.0, float(env.unwrapped.data.qpos[0])
            return observation, info

        def record_step(action):
            nonlocal total
            observation, reward, terminated, truncated, info = step(action)
            total += reward
            if terminated or truncated:
                raw_episodes.append((total, float(env.unwrapped.data.qpos[0]) - initial_x))
            return observation, reward, terminated, truncated, info

        env.reset, env.step = record_reset, record_step
        return env

    monkeypatch.setattr(ActorCritic, "get_action_and_value", check_action)
    monkeypatch.setattr(Normalizer, "normalize_observations", check_normalize)
    monkeypatch.setattr("gravity_adapt.evaluate.make_env", record_env)
    first = run_evaluation(
        checkpoint, seed=10000, environment=environment, episodes=3, output_dir=tmp_path,
    )
    second = run_evaluation(
        checkpoint, seed=10000, environment=environment, episodes=3, output_dir=tmp_path,
    )
    assert first.directory != second.directory
    assert (first.directory / "episodes.csv").read_bytes() == (
        second.directory / "episodes.csv"
    ).read_bytes()
    assert [args for args, _ in resets] == [{"seed": 10000}, {}, {}] * 2
    assert not np.array_equal(resets[0][1], resets[1][1])
    with (first.directory / "episodes.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(row["episode"]) for row in rows] == [1, 2, 3]
    for row, (reward, displacement) in zip(rows, raw_episodes[:3], strict=True):
        assert float(row["return"]) == pytest.approx(reward)
        assert float(row["x_displacement"]) == pytest.approx(displacement)
        assert int(row["length"]) == expected_environment.max_episode_steps
        assert row["truncated"] == "True"
    assert checkpoint.read_bytes() == original
    for model in models:
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())
    assert all(item.state_dict() == saved["normalization"] for item in normalizers)
    metadata = json.loads((first.directory / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["evaluation"] == {
        "checkpoint": str(checkpoint), "checkpoint_env_steps": 8, "episodes": 3, "explore": False,
        "video": False,
    }
    recorded = load_config(first.directory / "config.yaml")
    assert recorded.run.seed == 10000
    assert recorded.environment.gravity == expected_environment.gravity
    assert recorded.environment.max_episode_steps == expected_environment.max_episode_steps
    assert recorded.environment.num_envs == 1
    assert expected_environment.num_envs == 4


def test_interrupted_evaluation_records_status(checkpoint, tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ActorCritic, "get_action_and_value", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_evaluation(checkpoint, seed=10000, output_dir=tmp_path)
    metadata_path = next(tmp_path.glob("*/metadata.json"))
    assert json.loads(metadata_path.read_text())["status"] == "interrupted"


def test_video_records_each_episode_without_changing_results(checkpoint, tmp_path, monkeypatch):
    frames = []

    def render(self):
        assert self.render_mode == "rgb_array"
        frame = np.full((64, 64, 3), 20 * (len(frames) + 1), dtype=np.uint8)
        frames.append(frame)
        return frame

    monkeypatch.setattr(HumanoidEnv, "render", render)
    plain = run_evaluation(checkpoint, seed=10000, episodes=2, output_dir=tmp_path)
    recorded = run_evaluation(checkpoint, seed=10000, episodes=2, video=True, output_dir=tmp_path)
    assert not (plain.directory / "videos").exists()
    assert (plain.directory / "episodes.csv").read_bytes() == (
        recorded.directory / "episodes.csv"
    ).read_bytes()
    videos = sorted((recorded.directory / "videos").glob("*.mp4"))
    assert [path.name for path in videos] == [
        "evaluation-episode-0.mp4", "evaluation-episode-1.mp4",
    ]
    assert len(frames) == 6
    for index, path in enumerate(videos):
        with get_reader(path) as reader:
            assert reader.count_frames() == 3
            assert reader.get_meta_data()["fps"] == 67
            for frame_index in (0, 2):
                decoded = reader.get_data(frame_index)
                np.testing.assert_allclose(decoded, frames[index * 3 + frame_index], atol=3)
    metadata = json.loads((recorded.directory / "metadata.json").read_text())
    assert metadata["evaluation"]["video"] is True
