import torch
from torch import Tensor

from gravity_adapt.config import PPOConfig
from gravity_adapt.models import ActorCritic


@torch.no_grad()
def compute_gae(
    rollout: dict[str, Tensor], next_values: Tensor, gamma: float, gae_lambda: float
) -> tuple[Tensor, Tensor]:
    values = rollout["values"]
    terminated = rollout["terminated"]
    done = terminated | rollout["truncated"]
    deltas = rollout["rewards"] + gamma * next_values * ~terminated - values
    advantages = torch.empty_like(values)
    next_advantage = torch.zeros_like(values[0])
    for step in reversed(range(len(values))):
        next_advantage = deltas[step] + gamma * gae_lambda * ~done[step] * next_advantage
        advantages[step] = next_advantage
    return advantages, advantages + values


def update_ppo(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Tensor],
    config: PPOConfig,
) -> dict[str, float]:
    with torch.no_grad():
        next_values = model.critic(rollout["next_observations"])
        rollout["advantages"], rollout["returns"] = compute_gae(
            rollout, next_values, config.gamma, config.gae_lambda
        )
    batch = {name: tensor.flatten(0, 1) for name, tensor in rollout.items()}
    batch_size = len(batch["observations"])
    totals = {}
    for _ in range(config.update_epochs):
        indices = torch.randperm(batch_size, device=batch["observations"].device)
        for minibatch in indices.split(config.minibatch_size):
            _, log_probs, entropy, values = model.get_action_and_value(
                batch["observations"][minibatch], batch["actions"][minibatch]
            )
            advantages = batch["advantages"][minibatch]
            if len(minibatch) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
            log_ratio = log_probs - batch["log_probs"][minibatch]
            ratio = log_ratio.exp()
            clipped_ratio = ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = 0.5 * (values - batch["returns"][minibatch]).square().mean()
            entropy = entropy.mean()

            loss = policy_loss + config.critic_loss_weight * critic_loss - config.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                metrics = {
                    "policy_loss": policy_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "approx_kl": ((ratio - 1) - log_ratio).mean(),
                    "clip_fraction": ((ratio - 1).abs() > config.clip_coef).float().mean(),
                    "grad_norm": grad_norm,
                }
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0) + value * len(minibatch)
    return {
        name: (value / (batch_size * config.update_epochs)).item()
        for name, value in totals.items()
    }
