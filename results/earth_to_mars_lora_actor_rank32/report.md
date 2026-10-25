# Earth-to-Mars actor LoRA (rank 32)

Ten Earth-trained policies adapted with rank-32 LoRA on the actor and evaluated on Mars.
The critic was fully fine-tuned.
Configuration: [configs/mars_lora_actor_rank32.yaml](../../configs/mars_lora_actor_rank32.yaml).
Implementation: commit `6c5aa9a`. Experiment started on 2026-10-25.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-32 LoRA on all actor linear layers, including the output layer; scaling 1 (alpha = rank).
- Base actor weights and biases frozen; adapters, log_std, and the entire critic trainable.
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
| Critic | 155,393 | 155,393 |
| Total | 359,363 | 199,858 |

Trainable actor parameters comprise 44,448 adapter parameters and 17 log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 84/100 | 6476.7 | 941.4 | 24.90 |
| 43 | 56/100 | 6790.7 | 888.9 | 31.83 |
| 44 | 59/100 | 6148.8 | 886.3 | 23.91 |
| 45 | 90/100 | 7551.7 | 979.7 | 35.49 |
| 46 | 58/100 | 6198.6 | 893.6 | 24.26 |
| 47 | 77/100 | 7608.5 | 947.8 | 38.09 |
| 48 | 65/100 | 6699.7 | 837.8 | 33.46 |
| 49 | 92/100 | 7050.0 | 967.0 | 30.48 |
| 50 | 85/100 | 6483.9 | 946.2 | 24.53 |
| 51 | 93/100 | 6853.4 | 978.7 | 27.43 |

- 759/1,000 episodes reached the limit (75.9%).
- Mean return: 6786.2; mean episode length: 926.7; mean X displacement: 29.44 m.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
