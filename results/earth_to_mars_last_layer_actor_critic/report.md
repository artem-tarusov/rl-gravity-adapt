# Earth-to-Mars actor–critic last-layer adaptation

Only the output layers of the actor and critic, and the actor log standard deviations, are trained.
Configuration: [configs/mars_last_layer_actor_critic.yaml](../../configs/mars_last_layer_actor_critic.yaml).
Implementation: commit `e9249f0`. Experiment started on 2026-11-17.

## Protocol

- Humanoid-v5, Mars gravity `[0, 0, -3.71]` m/s²; episode limit 1,000 steps.
- Ten final Earth checkpoints trained for 5,000,000 transitions each, seeds 42–51.
- Each checkpoint adapted for 5,000,000 new Mars transitions with its original training seed.
- Separate actor and critic MLPs with two hidden layers of 256 units and Tanh.
- Output-layer weights and biases are inherited from Earth and trained; actor hidden layers remain frozen.
- Critic hidden layers remain frozen.
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
| Critic | 155,393 | 257 |
| Total | 314,915 | 4,643 |

The actor count includes 4,369 output-layer parameters and 17 log_std parameters.

## Evaluation

| Training seed | Reached limit | Mean return | Mean length | Mean X displacement (m) |
|---|---:|---:|---:|---:|
| 42 | 89/100 | 6146.8 | 955.8 | 19.95 |
| 43 | 58/100 | 5936.5 | 866.9 | 22.60 |
| 44 | 36/100 | 4838.7 | 766.0 | 14.96 |
| 45 | 77/100 | 6338.7 | 932.8 | 23.53 |
| 46 | 86/100 | 5700.5 | 946.7 | 14.93 |
| 47 | 77/100 | 6414.3 | 916.4 | 25.39 |
| 48 | 44/100 | 5047.1 | 726.5 | 19.86 |
| 49 | 91/100 | 6283.4 | 957.3 | 21.57 |
| 50 | 83/100 | 6027.0 | 941.8 | 19.22 |
| 51 | 90/100 | 6262.7 | 975.0 | 20.33 |

Aggregate values below are mean ± sample standard deviation across the ten policy-level means or rates, not across pooled episodes. Every policy has equal weight.

| Metric | Mean ± sample SD |
|---|---:|
| Return | 5899.6 ± 548.5 |
| Reached limit (%) | 73.1 ± 20.0 |
| Length (steps) | 898.5 ± 85.9 |
| X displacement (m) | 20.23 ± 3.37 |

- 731/1,000 evaluation episodes reached the limit; 269 terminated before it.
- Seed 49 has the highest time-limit rate (91/100); seed 47 has the highest mean return (6,414.3) and displacement (25.39 m).
- Seed 44 has the lowest mean return (4,838.7) and time-limit rate (36/100).
- Seed 48 has the shortest mean episode (726.5 steps) and largest within-policy return SD (2,460.0); its median return is 6,516.1.
- Within-policy return SD ranges from 617.9 (seed 51) to 2,460.0 (seed 48).

## Training diagnostics

Episodes are assigned to non-overlapping one-million-transition windows by their completion step. Each statistic is first averaged within each seed and then across the ten seeds. Training uses sampled actions; these returns are distinct from final mean-action evaluation.
PPO diagnostics use only full 8,192-transition rollouts. The final shortened rollout is excluded from these diagnostic averages.

| Transitions (millions) | Mean training return | Mean training length | Critic loss | Entropy | Approx. KL | Clip fraction | Grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 | 2714.9 | 456.4 | 0.01844 | 19.15 | 0.00727 | 0.1024 | 3.61 |
| 1–2 | 3716.8 | 603.1 | 0.01538 | 18.31 | 0.00630 | 0.0845 | 3.83 |
| 2–3 | 4215.1 | 672.1 | 0.01432 | 17.71 | 0.00502 | 0.0610 | 4.04 |
| 3–4 | 4466.4 | 700.9 | 0.01411 | 17.31 | 0.00320 | 0.0305 | 4.28 |
| 4–5 | 4667.8 | 728.1 | 0.01403 | 17.15 | 0.00076 | 0.0029 | 4.51 |

- Aggregate mean training return increases in each successive window. Eight seeds have higher window-average return at 4–5 million than at 3–4 million transitions; seeds 44 and 45 decrease by 25.0 and 43.6.
- Critic loss decreases and changes little over the final windows.
- Entropy, approximate KL, and clipping fraction decrease across the windows; the learning-rate schedule approaches zero near the end.
- Grad norm is recorded before gradient clipping. Its reported value can exceed the clipping threshold of 0.5.
- These diagnostics do not establish convergence or performance beyond the evaluated horizon.

## Validation and artifacts

All ten runs completed 5,000,000 transitions and 611 rollouts. All logged update metrics and final model weights are finite. Frozen weights match the respective Earth checkpoint exactly; all trainable parameter tensors changed.
All ten evaluations contain exactly 100 episodes. No episode is simultaneously terminated and truncated.

[Evaluation summary](summary.csv) · [Evaluation episodes](evaluation_episodes.csv) · [Training windows](training_windows.csv)

Summary rows identify source, adaptation, and evaluation runs. The return_std column is the sample standard deviation across each policy’s 100 evaluation episodes. Checkpoints and full training logs are stored locally and are not included in the repository.
