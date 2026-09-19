import json
from pathlib import Path

from gravity_adapt.config import ModelConfig, load_config
from gravity_adapt.smoke import run_smoke


def test_smoke_counts_real_transitions_across_resets(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs" / "earth.yaml"
    config = load_config(path)
    config.environment.num_envs = 2
    config.environment.max_episode_steps = 1
    config.model = ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8])
    config.run.name = "smoke_test"
    config.run.output_dir = str(tmp_path)
    run = run_smoke(config, steps=3)
    assert load_config(run.directory / "config.yaml") == config
    metadata = json.loads((run.directory / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["result"]["vector_steps"] == 3
    assert metadata["result"]["env_steps"] == 6
    assert metadata["result"]["completed_episodes"] == 6
    assert metadata["result"]["episode_lengths"] == [1] * 6
    assert metadata["result"]["observation_size"] == 348
