# Code archaeology audit of rlrmp and feedbax

This experiment builds an exhaustive, machine-crunchable census of every
top-level code object in rlrmp (`src/`, `tests/`, `scripts/`,
`results/*/scripts/`) and in the feedbax package it depends on, so that later
classification passes can judge purpose, generality, and usage status as
computed facts rather than impressions. The motivating question: rlrmp's
`src/` is roughly 115k lines across 148 modules, close in size to the ~112k
lines / 301 modules of the feedbax framework it is supposed to mostly consume;
existing `feedbax_contract` CI gates check that code behaves correctly at
registration/custody/import boundaries, but no gate asks whether a given piece
of code should exist at all, at its current location, in its current form.
Known symptoms that motivated filing this audit include issue-hash-named
run-row planners and hyperparameter bundles baked into `src/`
(`src/rlrmp/train/cs_nominal_gru.py` alone is ~8,900 lines), and objects that
look feedbax-general sitting in rlrmp. See `notes/audit_plan.md` for the full
phase plan, `notes/record_schema.md` for the fixed per-object classification
schema later phases will emit against, and `scripts/census.py` for the Phase 0
deterministic census/cross-reference builder. The corpus itself (JSONL
records, chunk plan, summary stats) is bulk output at
`_artifacts/05883e7/audit/census/` — regenerable by rerunning the script, not
tracked in git.
