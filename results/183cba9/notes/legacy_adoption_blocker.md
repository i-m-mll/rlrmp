# Legacy Baseline Adoption Blocker

Issue 183cba9 reached the local feedbax adoption tool, but the 08483d5 baseline
was not adopted because the LeafManifest dump cannot yet reconstruct the legacy
producer template in the available local Python environments.

## Confirmed Local Inputs

| item | value |
| --- | --- |
| legacy checkpoint | `_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_0012000` |
| checkpoint bytes | `model.eqx`, `optimizer_state.eqx`, `metadata.json` are present |
| completed batches | `12000` |
| requested producing commit | `9f919c65e52b0042181d615d4a40e1cc6fab5d0b` |
| tracked legacy spec | `results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json` |
| current adoption run spec | `results/3cd018b/runs/ramp3500_to1000.json` |
| feedbax TrainingRunSpec sidecar | `results/3cd018b/runs/ramp3500_to1000/feedbax_training_run_spec.json` |
| current slot summary | `results/3cd018b/notes/current_slot_summary.json` |

## Prepared Continuation Facts

The generated row uses adaptive-epsilon training with `n_train_batches=16500`.
The issue-local adoption context records `stop_after_batches=12500`, so a local
smoke continuation from the adopted `completed_batches=12000` checkpoint would
run `500` batches.  The current slot summary records
`schedule_start_batch=12000` and optimizer diagnostic target batches `16500`.

## Blocking Evidence

The feedbax adoption tool was invoked from:

`/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/feature__48b8b91-legacy-checkpoint-adoption`

The manifest dump command reached the temporary producing-commit checkout, but
the legacy producer imports old feedbax APIs and optional trainer dependencies
at module import time before the model/optimizer templates can be constructed.
Observed blockers, in order:

| attempt | blocker |
| --- | --- |
| default dump | missing `ruamel.yaml` in the environment selected by the temporary checkout |
| prepared feedbax env | old producer requires `feedbax.training.train`, absent from the current adoption branch |
| legacy feedbax API path | old trainer imports optional `tensorboardX` and SQLAlchemy surfaces |
| issue-local import shims | old feedbax config import next requires `plotly.graph_objects` |

No `leaf_manifest.json` was written, and no adopted Feedbax checkpoint custody
transaction was published.

## Next Action

Run the feedbax LeafManifest dump in a legacy-compatible environment for the
producer commit, or extend the feedbax dump tool so the temporary producing
checkout can resolve its historical dependency lock without mutating the current
adoption environment.  After a manifest exists, rerun:

```bash
PYTHONPATH="/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/feature__48b8b91-legacy-checkpoint-adoption:/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/feature__48b8b91-legacy-checkpoint-adoption/.venv/lib/python3.13/site-packages:/Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/feature__183cba9-legacy-checkpoint-backward-pass/src:/Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/feature__183cba9-legacy-checkpoint-backward-pass/results/183cba9/scripts" \
  uv run --no-sync python \
  "/Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/feature__183cba9-legacy-checkpoint-backward-pass/results/183cba9/scripts/adopt_08483d5_baseline.py"
```

