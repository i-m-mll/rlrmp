# Saturated Pre-Go Smoke Tests

Both smoke tests reused the previous best full-trial power-law cell but changed
only `nn_output_pre_go` and shortened warmup to 1000 batches on a secure RTX
5090.

| Cell | `nn_output_pre_go` | Final training loss | Final validation loss | Interpretation |
|---|---:|---:|---:|---|
| `smoke__full_trial_pl_prego100_1k` | 100.0 | `3.80e+00` | `6.33e+00` | Finite, but clearly worse than the prior `prego=1` early trajectory. |
| `smoke__full_trial_pl_prego10_1k` | 10.0 | `2.70e+00` | `5.19e+00` | Better than `100`, but still plateaued near the prior run's iteration-600 validation loss. |

The prior `full_trial_pl__prego_1` run reported validation loss `5.11e+00` at
iteration 600 and `1.65e+00` at iteration 1200. These smoke tests therefore do
not justify saturating the pre-go penalty to 10 or 100 for the main movement-ramp
matrix without another adjustment.

Co-authored-by: Codex (GPT-5) <codex@openai.com>
