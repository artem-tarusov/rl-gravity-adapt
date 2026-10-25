# Earth-to-Mars actor LoRA (rank 16)

Ten Earth-trained policies adapted with rank-16 LoRA on the actor and evaluated on Mars.
The critic was fully fine-tuned.
Configuration: [configs/mars_lora_actor_rank16.yaml](../../configs/mars_lora_actor_rank16.yaml).
Implementation: commit `6c5aa9a`. Experiment started on 2026-10-25.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy adapted for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Rank-16 LoRA on all actor linear layers, including the output layer; scaling 1 (alpha = rank).
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
| Actor | 181,746 | 22,241 |
| Critic | 155,393 | 155,393 |
| Total | 337,139 | 177,634 |

Trainable actor parameters comprise 22,224 adapter parameters and 17 log_std parameters.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 94/100 | 6662.6 | 981.9 | 24.71 |
| 43 | 49/100 | 6496.0 | 867.9 | 29.48 |
| 44 | 64/100 | 6067.2 | 891.1 | 22.57 |
| 45 | 67/100 | 6403.9 | 877.1 | 27.50 |
| 46 | 61/100 | 5664.8 | 854.3 | 20.06 |
| 47 | 82/100 | 7478.4 | 955.6 | 36.02 |
| 48 | 64/100 | 6517.5 | 842.8 | 30.95 |
| 49 | 98/100 | 7050.9 | 992.6 | 29.08 |
| 50 | 72/100 | 5810.3 | 852.5 | 21.65 |
| 51 | 87/100 | 6704.3 | 953.2 | 27.02 |

- 738/1,000 episodes reached the limit (73.8%).
- Mean return: 6485.6; mean episode length: 906.9; mean X displacement: 26.90 m.
- Each model has equal weight in the aggregate means.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, LoRA adaptation runs, and evaluation runs.
The return_std column is the sample standard deviation across the 100 episodes of each model.
Checkpoints include base weights, adapters, and normalization statistics and are not included in the repository.
