import math

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from gravity_adapt.config import ModelConfig
from gravity_adapt.lora import LoRALinear


def _mlp(
    input_size: int, hidden_sizes: list[int], output_size: int, output_gain: float
) -> nn.Sequential:
    layers = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        layer = nn.Linear(previous_size, hidden_size)
        nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.Tanh()])
        previous_size = hidden_size

    output_layer = nn.Linear(previous_size, output_size)
    nn.init.orthogonal_(output_layer.weight, gain=output_gain)
    nn.init.zeros_(output_layer.bias)
    return nn.Sequential(*layers, output_layer)


class Actor(nn.Module):
    """Gaussian actor; log_std is learned and independent of the observation."""

    def __init__(self, observation_size: int, action_size: int, config: ModelConfig):
        super().__init__()
        self.model = _mlp(
            observation_size, config.actor_hidden_sizes, action_size, output_gain=0.01
        )
        if config.lora_rank:
            for index, layer in enumerate(self.model):
                if isinstance(layer, nn.Linear):
                    self.model[index] = LoRALinear(layer, config.lora_rank)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, observations: Tensor) -> Independent:
        mean = self.model(observations)
        std = self.log_std.exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)


class Critic(nn.Module):
    def __init__(self, observation_size: int, config: ModelConfig):
        super().__init__()
        self.model = _mlp(observation_size, config.critic_hidden_sizes, 1, output_gain=1.0)

    def forward(self, observations: Tensor) -> Tensor:
        return self.model(observations).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, config: ModelConfig):
        super().__init__()
        self.actor = Actor(observation_size, action_size, config)
        self.critic = Critic(observation_size, config)

    def get_action_and_value(
        self, observations: Tensor, actions: Tensor | None = None, *, explore: bool = True
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        distribution = self.actor(observations)
        if actions is None:
            actions = distribution.sample() if explore else distribution.mean
        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(observations),
        )
