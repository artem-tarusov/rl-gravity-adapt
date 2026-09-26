# Earth baseline verification

Fixed on 2026-09-26 for subsequent gravity-adaptation experiments.
Use all three training seeds (42, 43, 44) as starting policies for comparisons.

Configuration: [configs/earth.yaml](../../configs/earth.yaml).
Training implementation reference: commit `9b08650`.

## Protocol

- Environment: Humanoid-v5, Earth gravity, 1,000-step episode limit.
- Training: 5,000,000 transitions, four synchronous environments, one PyTorch CPU thread.
- Initial learning rate: 1e-4. Linear decay enabled for training seeds 42, 43, and 44.
- Seeds 43 and 44 were trained from scratch in separate concurrent processes.
- All three models were evaluated without exploration on the same 100-episode sequence, seeded once with 20000. Observation normalization was frozen.
- Each selected checkpoint is the final update at 5,000,000 transitions.

## Results

| Training seed | Reached limit | Return mean ± SD | Return median | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|---:|
| 42 | 92/100 | 6542.7 ± 555.0 | 6686.6 | 979.9 | 24.77 |
| 43 | 86/100 | 6242.0 ± 1163.0 | 6634.2 | 939.9 | 24.14 |
| 44 | 84/100 | 5917.5 ± 1316.2 | 6408.5 | 923.6 | 20.63 |

## Findings

- The mean limit-reaching rate across the three seeds was 87.3% (262/300 episodes).
- Falls remain in 8–16% of episodes. Minimum evaluation lengths were 457, 199, and 122 steps for seeds 42, 43, and 44.
- This configuration and these checkpoints form the fixed Earth baseline.

## Limits

- There are three independent training runs. Evaluation episodes share a trained policy, and models use the same initial-state sequence.
- The test covers the standard Earth reset distribution and episode horizon, not perturbations, other gravity regimes, or longer episodes.
- Runtime comparisons are not controlled because seeds 43 and 44 trained concurrently.

## Artifacts

[Verification plots](verification.png) · [Full summary](summary.csv)

Checkpoints include model weights and normalization statistics and are not included
in the repository.

Training and evaluation run identifiers:

| Training seed | Training run | Evaluation run |
|---|---|---|
| 42 | `2026-09-26_10-12-47Z_earth_pilot_s42_9fcf9a8f` | `2026-09-26_10-46-07Z_earth_pilot_evaluation_s20000_5019ca47` |
| 43 | `2026-09-26_10-46-08Z_earth_pilot_s43_9c0165e5` | `2026-09-26_11-06-57Z_earth_pilot_evaluation_s20000_81f4ba3d` |
| 44 | `2026-09-26_10-46-09Z_earth_pilot_s44_ca023f26` | `2026-09-26_11-07-26Z_earth_pilot_evaluation_s20000_e46fc7c6` |
