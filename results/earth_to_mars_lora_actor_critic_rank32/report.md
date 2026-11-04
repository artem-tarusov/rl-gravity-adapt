# Earth-to-Mars actor and critic LoRA (rank 32)

Ten Earth-trained policies adapted with rank-32 LoRA on both actor and critic and evaluated on Mars.
Configuration: [configs/mars_lora_actor_critic_rank32.yaml](../../configs/mars_lora_actor_critic_rank32.yaml).
Implementation: commit `d0bcb09`. Experiment started on 2026-11-04.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Up to fifteen models trained concurrently across three ranks.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-32 LoRA on all actor and critic linear layers, including both output layers; scaling 1 (alpha = rank).
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
| Actor | 203,970 | 44,465 |
| Critic | 199,329 | 43,936 |
| Total | 403,299 | 88,401 |

Trainable parameters comprise 44,448 actor adapter parameters, 43,936 critic adapter parameters, and 17 actor log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 90/100 | 6636.8 | 967.7 | 25.25 |
| 43 | 49/100 | 6409.4 | 836.3 | 30.26 |
| 44 | 67/100 | 5977.2 | 880.4 | 22.09 |
| 45 | 90/100 | 7051.2 | 960.3 | 30.48 |
| 46 | 78/100 | 5924.2 | 910.5 | 19.83 |
| 47 | 85/100 | 7970.1 | 977.1 | 40.78 |
| 48 | 66/100 | 7455.3 | 898.4 | 39.30 |
| 49 | 98/100 | 6844.2 | 995.1 | 26.22 |
| 50 | 72/100 | 6437.3 | 919.5 | 25.53 |
| 51 | 87/100 | 6636.1 | 948.0 | 26.57 |

- 782/1,000 episodes reached the limit (78.2%).
- Mean return: 6734.2; mean episode length: 929.3; mean X displacement: 28.63 m.
- Sample standard deviation of the ten model mean returns: 632.6.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
