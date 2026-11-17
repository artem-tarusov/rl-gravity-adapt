from dataclasses import asdict
from pathlib import Path

import pytest

from gravity_adapt.config import ModelConfig, load_config, parse_config


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


@pytest.mark.parametrize("rank_key", ["actor_lora_rank", "lora_rank"])
def test_actor_lora_rank_loading_and_cli_override(tmp_path, rank_key):
    source = Path(__file__).resolve().parents[1] / "configs" / "mars_lora_actor_rank4.yaml"
    path = tmp_path / "config.yaml"
    legacy = source.read_text().replace("  actor_adaptation: lora\n", "")
    path.write_text(legacy.replace("actor_lora_rank", rank_key))
    assert load_config(path).model.actor_lora_rank == 4
    config = parse_config(["--config", str(path), "--model.actor-lora-rank", "16"])
    assert config.model.actor_lora_rank == 16
    assert config.model.critic_lora_rank == 0
    assert config.model.actor_adaptation == "lora"
    assert config.model.critic_adaptation == "full"
    overridden = parse_config([
        "--config", str(path), "--model.actor-adaptation", "last_layer",
    ])
    assert overridden.model.actor_adaptation == "last_layer"


@pytest.mark.parametrize("fields,actor,critic", [
    ({}, "full", "full"),
    ({"lora_rank": 4}, "lora", "full"),
    ({"actor_lora_rank": 4, "critic_lora_rank": 4}, "lora", "lora"),
    ({"actor_last_layer_only": True}, "last_layer", "full"),
    (
        {"actor_last_layer_only": True, "critic_last_layer_only": True}, "last_layer", "last_layer",
    ),
    ({"actor_lora_rank": 4, "actor_last_layer_only": False}, "lora", "full"),
    ({"actor_adaptation": "full", "actor_lora_rank": 4}, "full", "full"),
    ({"actor_adaptation": "full", "actor_last_layer_only": True}, "full", "full"),
])
def test_legacy_model_config_preserves_explicit_modes(fields, actor, critic):
    data = {"actor_hidden_sizes": [8], "critic_hidden_sizes": [8], **fields}
    original = dict(data)
    config = ModelConfig.from_dict(data)
    assert config.actor_adaptation == actor
    assert config.critic_adaptation == critic
    assert data == original
    assert not any(key.endswith("last_layer_only") for key in asdict(config))
    assert "lora_rank" not in asdict(config)
    assert ModelConfig.from_dict(asdict(config)) == config
