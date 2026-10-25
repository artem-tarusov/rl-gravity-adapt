# Earth-to-Mars actor LoRA (rank 4)

Ten Earth-trained policies adapted with rank-4 LoRA on the actor and evaluated on Mars.
The critic was fully fine-tuned.
Configuration: [configs/mars_lora_actor_rank4.yaml](../../configs/mars_lora_actor_rank4.yaml).
Implementation: commit `6c5aa9a`. Experiment started on 2026-10-25.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-4 LoRA on all actor linear layers, including the output layer; scaling 1 (alpha = rank).
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
| Actor | 165,078 | 5,573 |
| Critic | 155,393 | 155,393 |
| Total | 320,471 | 160,966 |

Trainable actor parameters comprise 5,556 adapter parameters and 17 log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 87/100 | 6405.3 | 976.6 | 21.91 |
| 43 | 50/100 | 4897.4 | 717.6 | 18.61 |
| 44 | 26/100 | 4687.4 | 734.9 | 14.79 |
| 45 | 51/100 | 6224.4 | 888.1 | 24.66 |
| 46 | 58/100 | 5136.0 | 829.6 | 14.90 |
| 47 | 73/100 | 6563.1 | 899.2 | 28.11 |
| 48 | 28/100 | 4643.2 | 651.8 | 19.26 |
| 49 | 99/100 | 6553.8 | 998.3 | 22.45 |
| 50 | 80/100 | 5929.0 | 933.4 | 18.41 |
| 51 | 89/100 | 6239.0 | 960.1 | 20.92 |

- 641/1,000 episodes reached the limit (64.1%).
- Mean return: 5727.9; mean episode length: 859.0; mean X displacement: 20.40 m.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
