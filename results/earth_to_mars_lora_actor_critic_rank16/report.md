# Earth-to-Mars actor and critic LoRA (rank 16)

Ten Earth-trained policies adapted with rank-16 LoRA on both actor and critic and evaluated on Mars.
Configuration: [configs/mars_lora_actor_critic_rank16.yaml](../../configs/mars_lora_actor_critic_rank16.yaml).
Implementation: commit `d0bcb09`. Experiment started on 2026-11-04.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Up to fifteen models trained concurrently across three ranks.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-16 LoRA on all actor and critic linear layers, including both output layers; scaling 1 (alpha = rank).
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
| Actor | 181,746 | 22,241 |
| Critic | 177,361 | 21,968 |
| Total | 359,107 | 44,209 |

Trainable parameters comprise 22,224 actor adapter parameters, 21,968 critic adapter parameters, and 17 actor log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 94/100 | 6632.7 | 983.6 | 24.27 |
| 43 | 76/100 | 6695.8 | 916.7 | 29.13 |
| 44 | 64/100 | 5885.7 | 895.6 | 20.06 |
| 45 | 60/100 | 6495.3 | 912.0 | 26.69 |
| 46 | 64/100 | 5595.1 | 859.6 | 18.82 |
| 47 | 93/100 | 7605.4 | 992.5 | 35.35 |
| 48 | 65/100 | 6644.7 | 848.3 | 32.33 |
| 49 | 99/100 | 6846.8 | 994.9 | 26.24 |
| 50 | 84/100 | 6384.7 | 932.4 | 24.01 |
| 51 | 80/100 | 6399.0 | 941.4 | 24.08 |

- 779/1,000 episodes reached the limit (77.9%).
- Mean return: 6518.5; mean episode length: 927.7; mean X displacement: 26.10 m.
- Sample standard deviation of the ten model mean returns: 541.1.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
