# Earth policies fine-tuned on Mars

Ten Earth-trained policies fully fine-tuned and evaluated on Mars.
Configuration: [configs/mars_finetune.yaml](../../configs/mars_finetune.yaml).
Implementation: commit `bcbda36`. Experiment started on 2026-10-11.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Source policies: final Earth checkpoints trained for 5,000,000 transitions with five PPO epochs, seeds 42–51.
- Each policy was fine-tuned for 5,000,000 new Mars transitions, using the same seed as its Earth training run.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Five PPO update epochs; learning rate 1e-4, linearly reduced toward zero over the fine-tuning budget.
- Actor, including log_std, and critic loaded from the source and fully trainable.
- Observation and return normalization statistics loaded from Earth and updated during fine-tuning; discounted-return accumulators reset.
- New Adam state, RNG initialized from the run seed, and counters starting at zero.
- Evaluation uses each final checkpoint, without checkpoint selection or further training.
- 100 episodes per model, evaluation seed 10000 on the first reset only.
- Mean actions, frozen fine-tuned observation normalization, raw rewards. All evaluations use the same initial-state sequence.
- Reached limit means truncation at 1,000 steps without simultaneous termination.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 90/100 | 7159.1 | 972.8 | 31.39 |
| 43 | 48/100 | 7027.7 | 862.4 | 36.42 |
| 44 | 78/100 | 6671.8 | 926.1 | 27.99 |
| 45 | 82/100 | 8084.8 | 949.9 | 43.75 |
| 46 | 67/100 | 7420.1 | 943.1 | 36.34 |
| 47 | 93/100 | 8522.0 | 965.5 | 48.14 |
| 48 | 76/100 | 7772.5 | 896.0 | 43.22 |
| 49 | 89/100 | 7229.4 | 955.2 | 33.48 |
| 50 | 73/100 | 6799.9 | 900.5 | 31.05 |
| 51 | 75/100 | 7114.4 | 919.2 | 34.28 |

- 771/1,000 episodes reached the limit (77.1%).
- Mean return: 7380.2; mean episode length: 929.1; mean X displacement: 36.60 m.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the source Earth training runs, Mars fine-tuning runs, and
evaluation runs. Checkpoints contain model weights and normalization statistics
and are not included in the repository.
