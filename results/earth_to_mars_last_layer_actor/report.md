# Earth-to-Mars actor last-layer adaptation

Only the actor output layer and log standard deviations are trained; the critic is fully fine-tuned.
Configuration: [configs/mars_last_layer_actor.yaml](../../configs/mars_last_layer_actor.yaml).
Implementation: commit `e9249f0`. Experiment started on 2026-11-17.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s²; episode limit 1,000 steps.
- Ten final Earth checkpoints trained for 5,000,000 transitions each, seeds 42–51.
- Each checkpoint adapted for 5,000,000 new Mars transitions with its original training seed.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Output-layer weights and biases are inherited from Earth and trained; actor hidden layers remain frozen.
- All critic weights and biases are trained.
- Actor log_std remains trainable. No layers are reinitialized and no adapters are added.
- Four synchronous environments and one PyTorch CPU thread per model; ten models trained concurrently.
- PPO: 2,048 rollout steps per environment, minibatches of 256, five update epochs.
- Learning rate 1e-4, linearly reduced toward zero; Adam epsilon 1e-5.
- Gamma 0.99; GAE lambda 0.95; clipping coefficient 0.2; entropy coefficient 0.
- Critic loss weight 0.5; maximum gradient norm 0.5.
- Observation and return normalization statistics inherited and updated; discounted-return accumulators reset.
- Fresh optimizer state, seed initialization, and transition counters for adaptation.
- Final checkpoints evaluated without intermediate checkpoint selection: 100 episodes each, seed 10000 on the first reset only.
- Mean actions, frozen checkpoint observation normalization, raw undiscounted rewards; shared initial-state sequence.
- Reached limit means truncation at 1,000 steps without simultaneous termination.

## Parameters

| Component | Total parameters | Trainable parameters |
|---|---:|---:|
| Actor | 159,522 | 4,386 |
| Critic | 155,393 | 155,393 |
| Total | 314,915 | 159,779 |

The actor count includes 4,369 output-layer parameters and 17 log_std parameters.

## Evaluation

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 92/100 | 6574.5 | 978.8 | 23.91 |
| 43 | 64/100 | 6008.4 | 860.5 | 23.80 |
| 44 | 60/100 | 5729.2 | 872.9 | 19.49 |
| 45 | 69/100 | 6624.4 | 912.7 | 28.12 |
| 46 | 69/100 | 5749.9 | 873.5 | 19.83 |
| 47 | 83/100 | 7436.3 | 965.3 | 34.92 |
| 48 | 35/100 | 5930.6 | 781.9 | 27.53 |
| 49 | 96/100 | 6801.9 | 990.5 | 26.09 |
| 50 | 78/100 | 5972.3 | 894.9 | 21.29 |
| 51 | 92/100 | 6637.1 | 976.7 | 24.84 |

Aggregate values below are mean ± sample standard deviation across the ten policy-level means or rates, not across pooled episodes. Every policy has equal weight.

| Metric | Mean ± sample SD |
|---|---:|
| Return | 6346.5 ± 555.1 |
| Reached limit (%) | 73.8 ± 18.5 |
| Length (steps) | 910.8 ± 67.1 |
| X displacement (m) | 24.98 ± 4.59 |

- 738/1,000 evaluation episodes reached the limit; 262 terminated before it.
- Seed 49 has the highest time-limit rate (96/100); seed 47 has the highest mean return (7,436.3) and displacement (34.92 m).
- Seed 48 has the lowest time-limit rate (35/100), shortest mean episode (781.9 steps), and largest within-policy return SD (2,134.0).
- Within-policy return SD ranges from 394.8 (seed 49) to 2,134.0 (seed 48).

## Training diagnostics

Episodes are assigned to non-overlapping one-million-transition windows by their completion step. Each statistic is first averaged within each seed and then across the ten seeds. Training uses sampled actions; these returns are distinct from final mean-action evaluation.
PPO diagnostics use only full 8,192-transition rollouts. The final shortened rollout is excluded from these diagnostic averages.

| Transitions (millions) | Mean training return | Mean training length | Critic loss | Entropy | Approx. KL | Clip fraction | Grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 | 2935.3 | 489.6 | 0.00888 | 19.15 | 0.00733 | 0.1035 | 3.57 |
| 1–2 | 4146.1 | 656.9 | 0.00758 | 18.36 | 0.00634 | 0.0851 | 3.75 |
| 2–3 | 4547.5 | 701.4 | 0.00786 | 17.84 | 0.00502 | 0.0612 | 3.96 |
| 3–4 | 4841.3 | 734.1 | 0.00821 | 17.51 | 0.00318 | 0.0305 | 4.19 |
| 4–5 | 5023.7 | 753.2 | 0.00947 | 17.38 | 0.00076 | 0.0030 | 4.44 |

- Mean training return increases in each successive window. All ten seeds have higher window-average return at 4–5 million than at 3–4 million transitions.
- Critic loss initially decreases, then rises modestly in later windows.
- Entropy, approximate KL, and clipping fraction decrease across the windows; the learning-rate schedule approaches zero near the end.
- Grad norm is recorded before gradient clipping. Its reported value can exceed the clipping threshold of 0.5.
- These diagnostics do not establish convergence or performance beyond the evaluated horizon.

## Validation and artifacts

All ten runs completed 5,000,000 transitions and 611 rollouts. All logged update metrics and final model weights are finite. Frozen weights match the respective Earth checkpoint exactly; all trainable parameter tensors changed.
All ten evaluations contain exactly 100 episodes. No episode is simultaneously terminated and truncated.

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv) · [Training windows](training_windows.csv)

Summary rows identify source, adaptation, and evaluation runs. The return_std column is the sample standard deviation across each policy’s 100 evaluation episodes. Checkpoints and full training logs are stored locally and are not included in the repository.
