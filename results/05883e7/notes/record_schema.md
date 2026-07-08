# Phase 2 classification record schema

This is the fixed schema Phase 2 (and Phase 4 verification) fan-out agents
emit, one JSON record per object id from
`_artifacts/05883e7/audit/census/objects.jsonl` (module-level records use the
corresponding row from `modules.jsonl`). Enums are verbatim — an agent that
needs a value outside an enum uses the enum's designated escape hatch
(`other` for `purpose`, `needs_decision` for `disposition`) plus a note; it
never invents a new label. Every field is required unless marked optional.

## Fields

- **`id`** (string): `<repo>:<module_relpath>:<qualname>`. `repo` is `rlrmp`
  or `feedbax`; `module_relpath` is the file path relative to its repo root
  (e.g. `src/rlrmp/train/cs_nominal_gru.py`, or
  `feedbax/feedbax/training/optimizer.py` for the feedbax package); `qualname`
  is the top-level object name (function, class, or module-level constant
  name). **Module records** (one per file, from `modules.jsonl`) use the
  literal qualname `__module__`, e.g.
  `rlrmp:src/rlrmp/train/cs_nominal_gru.py:__module__`.

- **`repo`** (enum): `rlrmp` | `feedbax`.

- **`tree`** (enum): `src` | `tests` | `scripts` | `results_scripts` |
  `package` (the feedbax package tree, i.e. `feedbax/feedbax/...`) |
  `examples` | `other`.

- **`kind`** (enum): `function` | `async_function` | `class` | `constant` |
  `module`.

- **`loc`** (int): line count of the object's span (or the whole file, for a
  module record).

- **`purpose`** (enum): `core_math_algorithm`, `model_graph_definition`,
  `training_loop`, `eval_logic`, `analysis_transform`,
  `spec_manifest_construction`, `registration_wiring`, `hp_or_data_constants`,
  `orchestration_cli`, `io_custody`, `viz`, `launch_resume_control`,
  `compat_legacy`, `test_support`, `typing_protocols`, `docs_meta`, `other`
  (requires a one-line note explaining why no other enum value fits).

- **`generality`** (enum):
  - `rlrmp_specific` — genuinely project/science-specific, correctly living
    in rlrmp.
  - `general_belongs_in_feedbax` — a general capability that should be a
    feedbax primitive but currently lives in rlrmp.
  - `duplicates_feedbax` — functionally reimplements something feedbax
    already provides.
  - `partial_overlap_feedbax` — overlaps a feedbax capability but adds
    genuine rlrmp-specific behavior; not a clean duplicate.
  - `framework_native` — a feedbax object correctly living in feedbax.
  - `project_specific_should_leave_feedbax` — a feedbax-reverse-audit hit: a
    feedbax object whose only real consumer is rlrmp and which should be
    demoted out of feedbax.

- **`feedbax_counterpart`** (string, dotted name, or `null`): the feedbax
  object this one duplicates/overlaps/should-become, per the Phase 1
  capability catalog. `null` if there is no counterpart.

- **`usage_status`** (enum): `live`, `test_only`, `legacy_only`, `dead`,
  `registry_or_string_referenced`, `ambiguous` — each paired with a required
  **`usage_evidence`** free-text field citing the actual census/xref facts
  (routed inbound counts, `registered`, `string_hits`, `in_all`,
  `unresolved_name_hits`) that back the verdict, not a restatement of the
  enum.

- **`contract_flags`** (array, subset of): `spec_first_violation`,
  `data_in_code`, `experiment_named_in_src`, `dangling_reference`,
  `custody_bypass`, `legacy_unbannered`, `misplaced_should_be_results_scripts`,
  `misplaced_should_be_library`, `none`. Use `[\"none\"]` (not an empty array)
  when no flag applies, so absence is never ambiguous with "not yet checked."

- **`disposition`** (enum): `keep`, `delete`, `move_to_feedbax`,
  `move_to_results_scripts`, `replace_with_declarative_surface`,
  `merge_dedupe`, `needs_decision` — with a required one-line
  **`disposition_note`**. `replace_with_declarative_surface` dispositions must
  identify which *governed spec surface* (manifest, recipe registration, run
  spec) is being imperatively hand-built where it should be authored
  declaratively — not merely propose swapping imperative control flow for
  feedbax's expression-AST grammar in place (that per-target retrofit path is
  closed; see `results/96ac0e5/notes/adjudication.md` /
  `results/86e1dd1/notes/post_grammar_adjudication.md`, and the "Cross-cutting
  constraint" section of `notes/audit_plan.md`).

- **`confidence`** (enum): `high` | `medium` | `low`.

- **`evidence`** (array of strings): `file:line` pointers backing the
  classification (e.g. `["src/rlrmp/train/cs_nominal_gru.py:2466-2519"]`).

## Worked example

`planned_ef9c882_start_pos_hold_rows` (`src/rlrmp/train/cs_nominal_gru.py:2466`)
is a known case: a module-level function that hardcodes the locked run rows
for experiment `ef9c882` — CLI argument lists, batch counts, loss-objective
selection — as Python literals, named after the tracking issue it was written
for, gated behind a `--planned-ef9c882-start-pos-hold-rows` CLI flag at line
5257 that just dumps its return value as JSON.

```json
{
  "id": "rlrmp:src/rlrmp/train/cs_nominal_gru.py:planned_ef9c882_start_pos_hold_rows",
  "repo": "rlrmp",
  "tree": "src",
  "kind": "function",
  "loc": 62,
  "purpose": "hp_or_data_constants",
  "generality": "rlrmp_specific",
  "feedbax_counterpart": null,
  "usage_status": "live",
  "usage_evidence": "Called from a CLI branch in the same module (cs_nominal_gru.py:5257-5258, `if args.planned_ef9c882_start_pos_hold_rows:`); no inbound refs from any other module (routed or unresolved); not registered; zero string_hits outside its own definition and CLI flag docstring.",
  "contract_flags": ["hp_or_data_constants", "spec_first_violation", "experiment_named_in_src"],
  "disposition": "delete",
  "disposition_note": "Locked run rows for a completed, already-run experiment (ef9c882) baked into src/ as Python literals and named after its tracking issue; per CLAUDE.md's 'Data stays separate from code' and script-placement rules this belongs in results/ef9c882/runs/*.json (or is simply dead now that the run happened), not as a callable in a general training module.",
  "confidence": "high",
  "evidence": ["src/rlrmp/train/cs_nominal_gru.py:2466-2519", "src/rlrmp/train/cs_nominal_gru.py:5257-5258"]
}
```

## Notes for chunk agents

- One record per object id in the assigned chunk; do not skip objects because
  they look trivial (a trivial object with `dead` usage_status is exactly the
  kind of finding this audit wants surfaced).
- `usage_evidence` and `evidence` must reference concrete census facts and
  `file:line` spans, not paraphrase the object's docstring.
- If an object is a `class`, classify the class as a whole; do not emit
  separate records for its methods (the census records method names/counts on
  the class row for context only).
- When `generality` is `duplicates_feedbax`, `feedbax_counterpart` is
  required (not `null`) and must name a real object from the Phase 1
  capability catalog.
