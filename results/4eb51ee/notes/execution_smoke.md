# A1 local engineering-smoke execution evidence

This packet is **non-scientific engineering-smoke evidence**. It does not answer the
cross-architecture certificate question.

## Outcome

The frozen six-row matrix authored, lowered, entered governed three-layer storage, and
validated successfully. The first local execution stopped before batch 1 at the shared
orchestration precondition `execute requires one common source checkpoint transaction`.
No training, checkpoint, evaluation, analysis, figure, or report artifact was produced.
Issue `52bacb3` owns this shared fresh-matrix execution defect and is structurally linked
to this experiment.

The emitter also produced an authored-matrix sidecar with a worktree-absolute URI.
Issue `e093cd9` owns portable repo-relative sidecar identity. The nonportable sidecar is
not tracked; the matrix SHA, artifact ID, and custody hashes remain recorded in the
structured execution status.

Targeted contract checks passed 11 tests covering the heterogeneous
authoring contract and the three-mode evaluation-manifest to analysis/report round trip.
They also verify that the child KPI input/report paths are trackable. These checks do
establish that the downstream mixed-mode renderer remains intact; they do
not substitute for this experiment's blocked 24 evaluations or final manifests.

The exact structured identities, planned run IDs, per-stage states, and log digest are in
`results/4eb51ee/runs/execution_status.json`. The raw shell-captured log is an issue-owned
local diagnostic at `_artifacts/4eb51ee/logs/sg_nominal_s42-stop50.log`; it is not a
Feedbax-custody artifact. Genuine Feedbax custody covers the resolved semantics snapshot
and execution capsule listed in the structured status.

## Commands

```bash
PYTHONPATH="$PWD/src" UV_CACHE_DIR=/tmp/uv-cache-4eb51ee PYTHONWARNINGS=ignore \
  uv run --no-sync python scripts/launch_training.py validate \
  results/4eb51ee/runs/matrix.json

PYTHONPATH="$PWD/src" UV_CACHE_DIR=/tmp/uv-cache-4eb51ee PYTHONWARNINGS=ignore \
  uv run --no-sync python scripts/launch_training.py dry-run \
  results/4eb51ee/runs/matrix.json

PYTHONPATH="$PWD/src" UV_CACHE_DIR=/tmp/uv-cache-4eb51ee \
  FEEDBAX_JAX_COMPILATION_CACHE_DIR=/tmp/feedbax-jax-cache-4eb51ee \
  PYTHONWARNINGS=ignore uv run --no-sync python scripts/launch_training.py execute \
  results/4eb51ee/runs/matrix.json --row sg_nominal_s42 --driver local \
  --stop-after-batches 50 --log-step 1

PYTHONPATH="$PWD/src" UV_CACHE_DIR=/tmp/uv-cache-4eb51ee \
  FEEDBAX_JAX_COMPILATION_CACHE_DIR=/tmp/feedbax-jax-cache-4eb51ee \
  scripts/dev_tests.sh tests/test_heterogeneous_training_matrix.py \
  tests/analysis/test_report_recipes.py::test_bridge_report_round_trips_three_certificate_modes

PYTHONPATH="$PWD/src" UV_CACHE_DIR=/tmp/uv-cache-4eb51ee \
  scripts/dev_tests.sh \
  tests/test_paths.py::test_experiment_marginal_cost_kpi_record_is_committable
```

## Verdicts

- Paved-road authoring and lowering: pass for all six frozen identities.
- Paved-road local execution: blocked before the first training batch.
- Early-training plausibility: not observed; there are no raw losses or movements.
- Scientific evidence: none.
- Escape pressure: nonzero because a fresh matrix cannot enter execution, but no bypass
  was used. The bypass inventory is empty.

The safe next action is to repair the owning orchestration contract, then rerun the exact
same matrix and command. It is not acceptable to invent a source checkpoint, call the
native executor directly, or hand-create downstream manifests.
