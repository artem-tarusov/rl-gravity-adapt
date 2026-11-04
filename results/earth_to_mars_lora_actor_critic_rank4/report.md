# Earth-to-Mars actor and critic LoRA (rank 4)

Ten Earth-trained policies adapted with rank-4 LoRA on both actor and critic and evaluated on Mars.
Configuration: [configs/mars_lora_actor_critic_rank4.yaml](../../configs/mars_lora_actor_critic_rank4.yaml).
Implementation: commit `d0bcb09`. Experiment started on 2026-11-04.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Up to fifteen models trained concurrently across three ranks.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-4 LoRA on all actor and critic linear layers, including both output layers; scaling 1 (alpha = rank).
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
| Actor | 165,078 | 5,573 |
| Critic | 160,885 | 5,492 |
| Total | 325,963 | 11,065 |

Trainable parameters comprise 5,556 actor adapter parameters, 5,492 critic adapter parameters, and 17 actor log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 88/100 | 6213.5 | 967.4 | 20.07 |
| 43 | 72/100 | 6001.0 | 881.7 | 22.46 |
| 44 | 60/100 | 5537.3 | 871.8 | 17.24 |
| 45 | 55/100 | 5999.7 | 862.7 | 23.46 |
| 46 | 67/100 | 5471.6 | 896.1 | 15.07 |
| 47 | 58/100 | 5889.5 | 835.8 | 23.64 |
| 48 | 30/100 | 5295.0 | 733.4 | 22.53 |
| 49 | 95/100 | 6295.8 | 979.6 | 20.37 |
| 50 | 73/100 | 5903.3 | 906.9 | 19.63 |
| 51 | 72/100 | 6001.0 | 906.2 | 21.20 |

- 670/1,000 episodes reached the limit (67.0%).
- Mean return: 5860.8; mean episode length: 884.2; mean X displacement: 20.57 m.
- Sample standard deviation of the ten model mean returns: 325.1.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
