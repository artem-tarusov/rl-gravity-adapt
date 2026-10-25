# Earth-to-Mars actor LoRA (rank 64)

Ten Earth-trained policies adapted with rank-64 LoRA on the actor and evaluated on Mars.
The critic was fully fine-tuned.
Configuration: [configs/mars_lora_actor_rank64.yaml](../../configs/mars_lora_actor_rank64.yaml).
Implementation: commit `6c5aa9a`. Experiment started on 2026-10-25.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-64 LoRA on all actor linear layers, including the output layer; scaling 1 (alpha = rank).
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
| Actor | 248,418 | 88,913 |
| Critic | 155,393 | 155,393 |
| Total | 403,811 | 244,306 |

Trainable actor parameters comprise 88,896 adapter parameters and 17 log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 93/100 | 7035.0 | 983.1 | 29.27 |
| 43 | 42/100 | 6331.9 | 826.9 | 29.88 |
| 44 | 75/100 | 6600.7 | 918.4 | 27.53 |
| 45 | 86/100 | 7455.8 | 961.6 | 35.38 |
| 46 | 89/100 | 7022.9 | 966.9 | 29.88 |
| 47 | 84/100 | 7913.3 | 945.3 | 41.95 |
| 48 | 76/100 | 7467.7 | 886.2 | 40.08 |
| 49 | 91/100 | 7086.7 | 955.0 | 31.78 |
| 50 | 74/100 | 6084.2 | 879.4 | 23.49 |
| 51 | 89/100 | 6953.0 | 960.1 | 29.89 |

- 799/1,000 episodes reached the limit (79.9%).
- Mean return: 6995.1; mean episode length: 928.3; mean X displacement: 31.91 m.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
