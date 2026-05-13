# Run Plan

Tracking issue: `b399efc`

## Smoke

Run the previous best pre-go matrix cell, `full_trial_pl__prego_1`, but change
`nn_output_pre_go` from `1.0` to `100.0` and shorten warmup to 1000 batches.
Proceed only if training is finite and the early loss curve is comparable to or
better than the previous 12k run's early trajectory, where validation loss was
about `1.65e+00` at iteration 1200.

## Matrix

Launch five warmup-only cells after the smoke passes. All cells use:

- `n_warmup_batches=12000`
- `n_adversary_batches=0`
- `n_replicates=5`
- `hidden_type=gru`
- `batch_size=250`
- `seed=42`
- `nn_output_pre_go=100.0`
- `effector_hold_pos=0.0`
- `effector_hold_vel=0.0`
- `effector_pos_running=1.0`
- `effector_pos_running_schedule=movement_ramp`
- `movement_ramp_duration_steps=60`
- no output jerk, no late position/velocity terms, no target-ratio update

Cells:

- `movement_ramp__linear`
- `movement_ramp__cosine`
- `movement_ramp__power2`
- `movement_ramp__power4`
- `movement_ramp__power6`

The movement ramp is zero before the movement epoch, rises from zero to one
over 60 fixed timesteps starting at movement onset, then remains one.

Co-authored-by: Codex (GPT-5) <codex@openai.com>
