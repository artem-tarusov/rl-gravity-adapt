# Third-party notices

These notices apply to third-party portions only. No project license is granted
for gravity-adapt's original contributions.

## CleanRL

Portions of the continuous-action PPO implementation are adapted from CleanRL:

| File | Adapted elements |
|---|---|
| `gravity_adapt/models.py` | Orthogonal initialization, Gaussian actor with learned log standard deviation, critic, and action/value interface |
| `gravity_adapt/ppo.py` | GAE, minibatch PPO updates, advantage normalization, losses, gradient clipping, and KL/clip diagnostics |
| `gravity_adapt/train.py` | Rollout collection, Adam configuration, and linear learning-rate decay |

`gravity_adapt/envs.py` and `gravity_adapt/normalization.py` also follow CleanRL's
action clipping and observation/reward normalization conventions. Running
statistics use Gymnasium's imported `RunningMeanStd`; its source is not vendored.

Upstream source reviewed: [cleanrl/ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/35896b1fefa9898b904f7e09bcbe6e168e15d2a9/cleanrl/ppo_continuous_action.py).
Upstream license reviewed: [LICENSE](https://github.com/vwxyzjn/cleanrl/blob/35896b1fefa9898b904f7e09bcbe6e168e15d2a9/LICENSE).

The following notice and license apply to the CleanRL-derived portions:

```text
MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
