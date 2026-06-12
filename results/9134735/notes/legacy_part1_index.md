# Legacy Part 1 Index

Issue `e6fee00` removed the live part1 training and analysis modules because
the frozen path was no longer a valid current construction path after
spec-first graph construction. Reproducing part1 should use a checkout from
before this removal, not the current working tree.

## Code Removed From Current Registration

- `src/rlrmp/modules/training/part1.py`
- `src/rlrmp/modules/analysis/part1/feedback_perts.py`
- `src/rlrmp/modules/analysis/part1/freq_response.py`
- `src/rlrmp/modules/analysis/part1/plant_perts.py`
- `src/rlrmp/modules/analysis/part1/unit_prefs.py`

The removed training and analysis modules were previously discoverable through
`rlrmp.register_experiment_package(..., parts=["part1", ...])` and part1 batch
entries under `src/rlrmp/config/batched/`.

## Legacy Analysis Names

| Analysis | Former module | Historical material |
|---|---|---|
| Feedback perturbations | `part1.feedback_perts` | `archive/1-analysis/1-2_feedback-perts.qmd`, manuscript part1 sections |
| Frequency response | `part1.freq_response` | `archive/1-analysis/1-3_freq-response.qmd` |
| Plant perturbations | `part1.plant_perts` | `archive/1-analysis/1-1_plant-perts.qmd`, `results/2ef67ca/` legacy archive |
| Unit preferences | `part1.unit_prefs` | `notebooks/markdown/part1__unit_prefs.md` |

## Replication Reference

No tag or branch was created in this worker lane. Issue `9134735` still owns the
explicit legacy tag/branch operation for frozen parts 1-3, and any tag creation
or push requires parent/user authorization.
