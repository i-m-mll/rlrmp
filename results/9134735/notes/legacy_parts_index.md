# Legacy Parts Index

Issue `9134735` tracks the end-of-program removal of the old "parts" mechanism
from the live working tree. The current cleanup is partial by design:

- Part1 was removed earlier under issue `e6fee00`.
- Part3 is removed by this lane because no current code imports it and its only
  live discoverability path was plugin registration.
- Part2 remains live because current training, eval, artifact migration,
  minimax, and analysis pipeline code still use
  `rlrmp.modules.training.part2.setup_task_model_pair`.

## Current Live State

`rlrmp.register_experiment_package(...)` now registers only `part2`. New
analysis work should use the declarative analysis bundle and capability-named
library surfaces introduced under umbrella `577806f`, not a new `part4`.

## Removed Code

Part1, removed by issue `e6fee00`:

- `src/rlrmp/modules/training/part1.py`
- `src/rlrmp/modules/analysis/part1/feedback_perts.py`
- `src/rlrmp/modules/analysis/part1/freq_response.py`
- `src/rlrmp/modules/analysis/part1/plant_perts.py`
- `src/rlrmp/modules/analysis/part1/unit_prefs.py`
- Part1 config entries under `src/rlrmp/config/modules/` and
  `src/rlrmp/config/batched/`

Part3, removed by this lane:

- `src/rlrmp/modules/analysis/part3/feedback_perts.py`
- `src/rlrmp/modules/analysis/part3/plant_perts.py`
- `src/rlrmp/modules/analysis/part3/__init__.py`

## Legacy Analysis Names

| Part | Analysis | Former module | Historical material |
|---|---|---|---|
| part1 | Feedback perturbations | `part1.feedback_perts` | `archive/1-analysis/1-2_feedback-perts.qmd`, manuscript part1 sections |
| part1 | Frequency response | `part1.freq_response` | `archive/1-analysis/1-3_freq-response.qmd` |
| part1 | Plant perturbations | `part1.plant_perts` | `archive/1-analysis/1-1_plant-perts.qmd`, `results/2ef67ca/` legacy archive |
| part1 | Unit preferences | `part1.unit_prefs` | `notebooks/markdown/part1__unit_prefs.md` |
| part2 | Feedback perturbations | `part2.feedback_perts` | `notebooks/markdown/part2__*.md`, `results/2ef67ca/`, `results/part2_5` legacy references, `_artifacts/part2_5` bulk references |
| part2 | Fixed points during reach | `part2.fps_reach` | `notebooks/markdown/part2__fps_reach.md` |
| part2 | Steady-state fixed points | `part2.fps_steady` | `notebooks/markdown/part2__fps_steady.md` |
| part2 | Plant perturbations | `part2.plant_perts` | `notebooks/markdown/part2__plant_perts.md` if present, `results/2ef67ca/` legacy archive |
| part2 | SISU perturbation | `part2.sisu_pert` | `TODO-analysis.md`, part2-era analysis notes |
| part2 | Tangling | `part2.tangling` | part2-era analysis notes |
| part2 | Unit perturbations | `part2.unit_perts` | `notebooks/markdown/part2__unit_perts.md`, `TODO-analysis.md` |
| part3 | Feedback perturbations | `part3.feedback_perts` | Frozen code only; use the pre-removal checkout for exact replication |
| part3 | Plant perturbations | `part3.plant_perts` | Frozen code only; use the pre-removal checkout for exact replication |

## Result And Artifact Locations

Known tracked legacy or narrative locations:

- `results/2ef67ca/` houses the legacy pre-migration archive, including older
  model configs.
- `notebooks/markdown/part1__unit_prefs.md`
- `notebooks/markdown/part2__fps_steady.md`
- `notebooks/markdown/part2__fps_reach.md`
- `notebooks/markdown/part2__unit_perts.md`
- `results/72fb8d9/synthesis.md` references part2_5-era run locations.
- `results/c723082/` contains part2_5-era induced-gain and flavor-b scripts and
  notes.
- `results/efc4d68/` contains baseline smoothness and variance-profile notes
  that reference part2_5-era artifacts.

Known ignored bulk locations referenced by tracked notes:

- `_artifacts/part2_5/`
- `_artifacts/part2_5/runpod/`
- `_artifacts/part2_5/runs/`

## Tag Guidance

No tag or branch was created in this lane. A single `legacy/parts-1-3` tag would
be ambiguous from the current branch because part1 was already removed before
this lane started and part2 still remains live.

Known safe reference points:

- Last known commit before part1 removal: `27d5a89^` contains
  `src/rlrmp/modules/training/part1.py`.
- Last known commit before part3 removal on this branch: `a952bcf` contains
  `src/rlrmp/modules/analysis/part3/`.
- Part2 has no final legacy-retirement target yet because it is still required
  by current code.

If a durable legacy reference is needed later, create narrowly named local tags
for these epochs and push only after explicit user authorization.
