# Earth policies evaluated on Mars

Ten Earth-trained policies evaluated on Mars on 2026-10-11, without further training.
Training configuration: [configs/earth.yaml](../../configs/earth.yaml).

## Protocol

- Training seeds 42–51; each policy was trained on Earth for 5,000,000 transitions with five PPO update epochs.
- Evaluation uses Humanoid-v5 with gravity `[0, 0, -3.71]` m/s² and a 1,000-step episode limit.
- 100 episodes per model, evaluation seed 10000 on the first reset only.
- Mean actions, unchanged model weights, frozen Earth observation normalization, raw rewards.
- Reached limit means truncation at 1,000 steps without simultaneous termination.

## Results

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 12/100 | 3240.9 | 536.8 | 8.73 |
| 43 | 0/100 | 1710.3 | 300.8 | 3.60 |
| 44 | 1/100 | 2519.0 | 420.2 | 6.65 |
| 45 | 14/100 | 3396.1 | 566.5 | 8.84 |
| 46 | 3/100 | 2634.3 | 455.3 | 6.01 |
| 47 | 11/100 | 2394.1 | 398.3 | 6.24 |
| 48 | 0/100 | 1565.6 | 273.8 | 3.39 |
| 49 | 6/100 | 1865.4 | 335.5 | 3.38 |
| 50 | 0/100 | 1574.3 | 285.6 | 2.80 |
| 51 | 4/100 | 3299.1 | 531.0 | 9.94 |

- 51/1,000 episodes reached the limit (5.1%).
- Mean return: 2419.9; mean length: 410.4; mean X displacement: 5.96 m.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

Summary rows identify the original Earth training runs and the new Mars
evaluation runs. Checkpoints are not included in the repository.
