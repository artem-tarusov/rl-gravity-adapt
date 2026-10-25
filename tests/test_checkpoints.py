from pathlib import Path

import numpy as np
import pytest
import torch

from gravity_adapt.checkpoints import load_checkpoint, save_checkpoint
from gravity_adapt.config import ModelConfig, load_config
from gravity_adapt.models import ActorCritic
from gravity_adapt.normalization import Normalizer


@pytest.mark.parametrize("actor_lora_rank", [0, 4])
def test_checkpoint_resume_and_finetune_restore_selected_state(
    tmp_path, monkeypatch, actor_lora_rank
):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "earth.yaml")
    config.model = ModelConfig(
        actor_hidden_sizes=[8], critic_hidden_sizes=[8], actor_lora_rank=actor_lora_rank,
    )
    config.environment.num_envs = 2
    torch.manual_seed(42)
    model = ActorCritic(3, 2, config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.learning_rate, eps=1e-5)
    normalizer = Normalizer(3, 2, config.ppo.gamma)
    normalizer.normalize_observations(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
    normalizer.normalize_rewards(np.array([1.0, 2.0]), np.array([False, False]))

    def step(model, optimizer):
        observations = torch.randn(4, 3)
        _, log_probs, entropy, values = model.get_action_and_value(observations)
        loss = -log_probs.mean() - 0.1 * entropy.mean() + (values - 1).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    step(model, optimizer)
    progress = {
        "env_steps": 4,
        "rollouts": 1,
        "completed_episodes": 0,
        "elapsed_seconds": 2.0,
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, normalizer, config, progress)
    legacy = torch.load(path, weights_only=True)
    legacy["config"]["model"].pop("actor_lora_rank")
    if actor_lora_rank:
        legacy["config"]["model"]["lora_rank"] = actor_lora_rank
    legacy["config"]["model"].pop("critic_lora_rank")
    torch.save(legacy, path)
    saved_normalization = normalizer.state_dict()
    step(model, optimizer)
    expected_random = torch.rand(3)
    restored = ActorCritic(3, 2, config.model)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1.0)
    restored_normalizer = Normalizer(3, 2, gamma=0.5)
    assert load_checkpoint(path, restored, restored_optimizer, restored_normalizer, config, mode="resume") == progress
    assert restored_normalizer.state_dict() == saved_normalization
    assert restored_optimizer.param_groups[0]["lr"] == config.ppo.learning_rate
    step(restored, restored_optimizer)
    torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, restored_optimizer.state_dict()["state"][key][name], rtol=0, atol=0
            )

    config.environment.gravity = [0.0, 0.0, -3.71]
    with pytest.raises(ValueError, match="environment"):
        load_checkpoint(path, restored, restored_optimizer, restored_normalizer, config, mode="resume")
    config.environment.gravity = [0.0, 0.0, -9.81]
    config.ppo.total_steps = 4
    with pytest.raises(ValueError, match="total_steps"):
        load_checkpoint(path, restored, restored_optimizer, restored_normalizer, config, mode="resume")
    original = path.read_bytes()

    config.environment.num_envs = 3
    config.ppo.gamma = 0.95
    config.run.seed = 7
    finetuned = ActorCritic(3, 2, config.model)
    fresh_optimizer = torch.optim.Adam(finetuned.parameters(), lr=2e-5)
    fresh_normalizer = Normalizer(3, 3, config.ppo.gamma)
    rng = torch.get_rng_state()
    assert load_checkpoint(path, finetuned, fresh_optimizer, fresh_normalizer, config, mode="finetune") is None
    for name, value in torch.load(path, weights_only=True)["model"].items():
        torch.testing.assert_close(value, finetuned.state_dict()[name], rtol=0, atol=0)
    assert fresh_optimizer.state_dict()["state"] == {}
    assert fresh_optimizer.param_groups[0]["lr"] == 2e-5
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert fresh_normalizer.state_dict() == {
        **saved_normalization, "discounted_returns": [0.0, 0.0, 0.0], "gamma": 0.95,
    }
    assert path.read_bytes() == original

    def interrupted_save(checkpoint, temporary):
        temporary.write_bytes(b"partial checkpoint")
        raise KeyboardInterrupt

    monkeypatch.setattr("gravity_adapt.checkpoints.torch.save", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(path, model, optimizer, normalizer, config, progress)
    assert path.read_bytes() == original
    assert torch.load(path, weights_only=True)["progress"] == progress
