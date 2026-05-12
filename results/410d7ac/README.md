# 410d7ac — Linear regulator vs tracker decoupling acid test (MVP)

MVP-scoped first-signal experiment under parent issue `d448c9d` and phase
umbrella `f695729`. Tests whether the **tracker** controller parameterisation
`u = u_ff(t) - K_t · e_t` decouples open-loop feedforward drive from
closed-loop feedback stiffness, relative to the pure **regulator**
`u = -K_t · e_t`, under adversarial training.

## Δv as a training-method comparison

Δv is the **peak forward velocity inflation between an adversarially-trained
model and the warmup-only baseline of the same architecture**:

    Δv_arch = (peak_v(arch_adversarial) - peak_v(arch_baseline)) / peak_v(arch_baseline)

This is a training-method comparison, *not* a test-time perturbation response.
Both models are evaluated at the same conditions; only the training procedure
differs. The discriminator prediction from `d448c9d` is `Δv_regulator > 0` and
`Δv_tracker ≈ 0`.

The prior MVP (commit `20ae797`, retracted on `410d7ac` comment 4) trained
warmup-only models and measured test-time perturbation response, which is a
different and uninteresting quantity. This corrected MVP trains **four**
models — baseline + adversarial for each of the two architectures — and
computes Δv as defined above.

## Layout

- `runs/linear_regulator__baseline/run.json` — regulator, n_adversary_batches=0
- `runs/linear_regulator__adversarial/run.json` — regulator, +500 adversarial batches
- `runs/linear_tracker__baseline/run.json` — tracker, n_adversary_batches=0
- `runs/linear_tracker__adversarial/run.json` — tracker, +500 adversarial batches
- `notes/decoupling_acid_test_mvp.md` — headline finding + discussion
- `notes/delta_v_summary.json` — machine-readable Δv table written by the analysis
- `figures/delta_v_signature/spec.json` — figure spec + auto-mirrored render

The legacy two-model specs (`runs/linear_regulator.json`,
`runs/linear_tracker.json`) from the retracted 20ae797 MVP were deleted as
part of this cleanup; the retraction comment on issue `410d7ac` is the
canonical historical record of the prior framing. The trained warmup models
from the prior MVP remain under their `_artifacts/` paths but are not
referenced by the corrected analysis (which loads its own pairs from the four
`<variant>/` directories above).

## Implementation

Architectures live in `src/rlrmp/networks/linear_controllers.py`. The training
script `scripts/train_minimax.py` gains `--hidden-type linear` and
`--hidden-type linear_tracker`; `setup_task_model_pair` dispatches to a
`point_mass_linear_controller` body builder that swaps `SimpleStagedNetwork`
for the LTV controllers without touching feedbax. Time is tracked implicitly
via a per-step counter the controller maintains in its `NetworkState.hidden`
channel (no `task.add_input` plumbing was needed).

The analysis script `scripts/analyse_linear_decoupling_mvp.py` loads all four
trained models, computes per-replicate Δv (paired by replicate index, since
the seed matches across each baseline/adversarial pair), and produces both a
stdout report and the `delta_v_signature` figure.

## Resumability

All four runs are resumable. Baselines (warmup-only) can be extended by
re-running with a larger `--n-warmup-batches`; adversarial runs support full
mid-training resumption via the `--resume` flag (the training script saves a
checkpoint every 100 batches to `<output-dir>/checkpoints_adversarial/`
containing model + adversaries + optimizer states + batch index + loss
histories). See each `run.json`'s `resumption_strategy` field for details.

See `notes/decoupling_acid_test_mvp.md` for the headline finding.
