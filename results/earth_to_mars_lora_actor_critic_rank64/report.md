# Earth-to-Mars actor and critic LoRA (rank 64)

Ten Earth-trained policies adapted with rank-64 LoRA on both actor and critic and evaluated on Mars.
Configuration: [configs/mars_lora_actor_critic_rank64.yaml](../../configs/mars_lora_actor_critic_rank64.yaml).
Implementation: commit `63bbc09`. Experiment started on 2026-10-25.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-64 LoRA on all actor and critic linear layers, including both output layers; scaling 1 (alpha = rank).
- Base weights and biases of both networks frozen; actor and critic adapters and actor log_std trainable.
- Adapter A initialized with Kaiming uniform using a = sqrt(5); B initialized to zero.
- Rollouts: 2,048 steps per environment; minibatches: 256; five PPO update epochs.
- Learning rate: 1e-4, linearly reduced toward zero over the adaptation budget.
- Gamma: 0.99; GAE lambda: 0.95; PPO clip coefficient: 0.2; entropy coefficient: 0.
- Critic loss weight: 0.5; maximum gradient norm: 0.5.
- Observation and return normalization statistics loaded from Earth and updated during adaptation; discounted-return accumulators reset.
- New Adam state, RNG initialized from the run seed, and counters starting at zero.
- Evaluation uses each final checkpoint, without checkpoint selection or further training.
- 100 episodes per model, evaluation seed 10000 on the first reset only.
- Mean actions, frozen adapted observation normalization, raw rewards. All evaluations use the same initial-state sequence.
- Reached limit means truncation at 1,000 steps without simultaneous termination.

## Parameters

| Component | Total parameters | Trainable parameters |
|---|---:|---:|
| Actor | 248,418 | 88,913 |
| Critic | 243,265 | 87,872 |
| Total | 491,683 | 176,785 |

Trainable parameters comprise 88,896 actor adapter parameters, 87,872 critic adapter parameters, and 17 actor log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 89/100 | 6704.3 | 964.2 | 26.40 |
| 43 | 60/100 | 6513.3 | 859.8 | 30.11 |
| 44 | 56/100 | 5905.2 | 838.4 | 23.67 |
| 45 | 69/100 | 7025.1 | 903.5 | 33.67 |
| 46 | 52/100 | 6178.2 | 866.7 | 25.52 |
| 47 | 88/100 | 8086.8 | 971.4 | 42.58 |
| 48 | 59/100 | 7013.3 | 853.0 | 36.51 |
| 49 | 85/100 | 6992.9 | 935.9 | 31.62 |
| 50 | 81/100 | 6792.2 | 937.8 | 28.70 |
| 51 | 89/100 | 7115.4 | 967.3 | 31.37 |

- 728/1,000 episodes reached the limit (72.8%).
- Mean return: 6832.7; mean episode length: 909.8; mean X displacement: 31.02 m.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
