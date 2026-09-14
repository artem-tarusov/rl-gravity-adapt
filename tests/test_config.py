from pathlib import Path

import pytest

from gravity_adapt.config import load_config, parse_config


def test_cli_overrides_yaml_and_preserves_other_values():
    path = Path(__file__).resolve().parents[1] / "configs" / "earth.yaml"
    config = parse_config(["--config", str(path), "--run.seed", "7", "--environment.num-envs", "2"])
    assert config.run.seed == 7
    assert config.environment.num_envs == 2
    assert config.run.name == "earth_baseline"
    assert config.model.actor_hidden_sizes == [256, 256]


def test_unknown_configuration_field_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("environment: {num_env: 4}")
    with pytest.raises(TypeError, match="num_env"):
        load_config(path)


def test_ppo_overrides():
    path = Path(__file__).resolve().parents[1] / "configs" / "earth.yaml"
    config = parse_config(["--config", str(path), "--ppo.total-steps", "24", "--ppo.gamma", "0.9"])
    assert config.ppo.total_steps == 24
    assert config.ppo.gamma == 0.9
    assert config.ppo.rollout_steps == 2048
