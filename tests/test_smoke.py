import json

from gravity_adapt.config import Config, EnvironmentConfig, ModelConfig, RunConfig, load_config
from gravity_adapt.smoke import run_smoke


def test_smoke_counts_real_transitions_across_resets(tmp_path):
    config = Config(
        environment=EnvironmentConfig(gravity=[0.0, 0.0, -9.81], num_envs=2, max_episode_steps=1),
        model=ModelConfig(actor_hidden_sizes=[8], critic_hidden_sizes=[8]),
        run=RunConfig(
            name="smoke_test",
            seed=42,
            device="cpu",
            output_dir=str(tmp_path),
            torch_threads=1,
        ),
    )
    run = run_smoke(config, steps=3)
    assert load_config(run.directory / "config.yaml") == config
    metadata = json.loads((run.directory / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["result"]["vector_steps"] == 3
    assert metadata["result"]["environment_steps"] == 6
    assert metadata["result"]["completed_episodes"] == 6
    assert metadata["result"]["episode_lengths"] == [1] * 6
    assert metadata["result"]["observation_size"] == 348
