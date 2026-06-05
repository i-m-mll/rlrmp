# Training Perturbation Mixture Audit

Issue: `1ad3c16`. Related coordination: `c99ad9d`.

## Current training mixture

The fixed-target perturbation-training path is implemented in
`src/rlrmp/train/cs_perturbation_training.py` and is enabled from
`scripts/train_cs_nominal_gru.py` via `--perturbation-training`.

For each training trial, membership is sampled from the supplied PRNG key:

| lane | default fraction | emitted perturbation |
|---|---:|---|
| nominal | 0.45 | no explicit perturbation beyond ordinary stochastic runtime |
| single family | 0.45 | one family sampled uniformly from `initial_position`, `initial_velocity`, `process_epsilon`, `command_input`, `sensory_feedback`, `delayed_observation` |
| mild combined | 0.10 | `initial_position` plus `command_input`, both scaled by `combined_amplitude_scale = 0.5` |

Within active single-family trials, signs and components are randomized. Initial
position and velocity perturb one random x/y component of
`mechanics.vector` at `t=0`. Process epsilon pulses perturb one random epsilon
component over a random start time. Command-input pulses perturb one random
command component over a random start time through the additive graph adapter.
Sensory-feedback and delayed-observation offsets perturb one random 4D channel
component; the current configuration passes full-trial duration to the random
pulse helper. Active families also sample `amplitude_level` from `[0.5, 1.0]`.

Default raw base amplitudes are:

| family | raw base amplitude | unit |
|---|---:|---|
| initial_position | 0.01 | m |
| initial_velocity | 0.05 | m/s |
| process_epsilon | 0.01 | epsilon |
| command_input | 1.0 | N |
| sensory_feedback | 0.01 | m or m/s channel units |
| delayed_observation | 0.01 | m or m/s channel units |

Validation bins are family-separated deterministic probes. They are not a replay
of the full training mixture; they expose nominal, single-family, and
mild-combined bins for selection/reporting.

## Interpretation guardrail

Perturbation uncertainty level is an experimental factor distinct from physical
perturbation amplitude. Increasing the breadth of families, signs, components,
timings, or mixture combinations can induce robustness instead of merely testing
ordinary feedback control. Raw-amplitude equality across families is not
physical-effect equality.

Future run specs now include
`hps.perturbation_training.mixture_semantics` so this distinction is attached to
new perturbation-training rows.

## Calibration baseline recommendation

The calibration arm should use:

1. nominal open-loop extLQG command replay for defining physical-effect bins by
   peak `delta x`;
2. closed-loop extLQG at those same amplitudes;
3. a nominal-only vanilla C&S GRU at those same amplitudes.

Perturbation-trained GRUs should be evaluated later as outcome models, not used
as the calibration baseline.

## Existing nominal-only GRU candidates

Best current local candidate: `5f70333`.

| experiment/run | artifact | status |
|---|---|---|
| `5f70333/lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | `_artifacts/5f70333/runs/lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64/trained_model.eqx` | 12k batches complete; nominal-only; full analytical Q/R/Q_f; selected validation/deterministic extLQG ratio 0.999723; clean action mismatch 0.005208 |
| `5f70333/lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64` | `_artifacts/5f70333/runs/lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64/trained_model.eqx` | 12k batches complete; nominal-only; full analytical Q/R/Q_f; selected validation/deterministic extLQG ratio 0.994077; clean action mismatch 0.000120 |

Older nominal-only checkpoints exist under `3b2af27` and `30f2313`, but their
standard-certificate action mismatch is much worse than `5f70333` in the
available notes. The `5f70333` rows are therefore the preferred calibration-GRU
baseline unless a fresh nominal-only row is deliberately trained.

No new training was launched for this audit.
