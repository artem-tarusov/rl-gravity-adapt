# Mars baseline

Ten policies trained from scratch on Mars.
Configuration: [configs/mars.yaml](../../configs/mars.yaml).
Training implementation: commit `3c91809`. Experiment started on 2026-10-11.

## Protocol

- Humanoid-v5, gravity `[0, 0, -3.71]` m/s², 1,000-step episode limit.
- Training seeds 42–51, 5,000,000 transitions per model, five PPO update epochs.
- Four synchronous environments and one PyTorch CPU thread per model. Ten models trained concurrently.
- Initial learning rate 1e-4, linearly reduced toward zero over the full budget.
- Evaluation uses each final checkpoint, with no checkpoint selection or further training.
- 100 episodes per model, evaluation seed 10000 on the first reset only.
- Mean actions, frozen observation normalization, raw rewards. All models use the same initial-state sequence.
- Reached limit means truncation at 1,000 steps without simultaneous termination.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 19/100 | 5336.7 | 728.3 | 23.49 |
| 43 | 18/100 | 4160.8 | 633.2 | 14.38 |
| 44 | 6/100 | 4060.5 | 628.3 | 13.46 |
| 45 | 37/100 | 5360.5 | 769.4 | 21.37 |
| 46 | 98/100 | 7380.4 | 994.5 | 32.85 |
| 47 | 37/100 | 4969.6 | 746.4 | 17.69 |
| 48 | 73/100 | 6173.7 | 916.9 | 22.42 |
| 49 | 75/100 | 6654.8 | 930.3 | 27.51 |
| 50 | 5/100 | 3968.2 | 603.6 | 13.85 |
| 51 | 14/100 | 4509.7 | 666.9 | 16.92 |

- 382/1,000 episodes reached the limit (38.2%).
- Mean return: 5257.5; mean episode length: 761.8; mean X displacement: 20.39 m.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the training and evaluation runs. Checkpoints contain model
weights and normalization statistics and are not included in the repository.
