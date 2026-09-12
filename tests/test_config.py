import pytest

from gravity_adapt.config import load_config, parse_config


def test_cli_overrides_yaml_and_preserves_other_values(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """\
environment:
  gravity: [0.0, 0.0, -3.71]
  num_envs: 3
  max_episode_steps: 500
model:
  actor_hidden_sizes: [16, 8]
  critic_hidden_sizes: [32]
run:
  name: pilot
  seed: 11
  device: cpu
  output_dir: runs
  torch_threads: 1
"""
    )
    config = parse_config(["--config", str(path), "--run.seed", "7", "--environment.num-envs", "2"])
    assert config.run.seed == 7
    assert config.environment.num_envs == 2
    assert config.run.name == "pilot"
    assert config.model.actor_hidden_sizes == [16, 8]


def test_unknown_configuration_field_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("environment: {num_env: 4}")
    with pytest.raises(TypeError, match="num_env"):
        load_config(path)
