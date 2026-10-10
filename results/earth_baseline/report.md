# Earth baseline

Ten policies trained from scratch on Earth and evaluated on 2026-10-11.

Configuration: [configs/earth.yaml](../../configs/earth.yaml).
The recorded models were trained with implementation commit `e69a71b` and five
update epochs. Their original `earth_epochs5` run identifiers are preserved in
the summary CSV file.

## Protocol

- Humanoid-v5, gravity `[0, 0, -9.81]` m/s², 1,000-step episode limit.
- Ten models trained from scratch, seeds 42–51, 5,000,000 transitions each.
- Four synchronous environments, CPU, one PyTorch thread per model.
- Separate actor and critic MLPs, two hidden layers of 256 units with Tanh.
- Rollouts: 2,048 steps per environment (8,192 transitions); minibatches: 256; update epochs: 5.
- Learning rate: 1e-4, linearly reduced toward zero over the full budget.
- Gamma: 0.99; GAE lambda: 0.95; PPO clip coefficient: 0.2.
- Entropy coefficient: 0; critic loss weight: 0.5; maximum gradient norm: 0.5.
- Every evaluation uses the final checkpoint, without checkpoint selection.
- This report uses 100 evaluation episodes per model with seed 10000.
  The seed is set only on the first reset of each sequence.
- Evaluation uses mean actions and frozen observation normalization. Reported
  returns use raw rewards. The same sequence is used for all ten models.
- Reached limit means truncation at 1,000 steps without simultaneous termination.

## Results

| Training seed | Reached limit | Mean return |
|---|---:|---:|
| 42 | 99/100 | 6383.7 |
| 43 | 90/100 | 6352.2 |
| 44 | 98/100 | 6232.7 |
| 45 | 99/100 | 6888.2 |
| 46 | 98/100 | 6398.7 |
| 47 | 100/100 | 7286.7 |
| 48 | 95/100 | 7117.8 |
| 49 | 100/100 | 6408.6 |
| 50 | 91/100 | 6276.7 |
| 51 | 100/100 | 6413.0 |

- 970/1,000 episodes reached the limit (97.0%); mean return 6575.8.
  Each model has equal weight.

## Artifacts

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv)

The CSV files contain only the selected five-epoch configuration. The summary file
identifies the original training and evaluation runs. Checkpoints include model
weights and normalization statistics and are not included in the repository.
