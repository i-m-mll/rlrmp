Confirmed. Both `output_feedback_phase_modulated_recurrent.py` (3221 lines) and `cs_stochastic_phase1.py` (787 lines) carry module-level `LEGACY (frozen 2026-07-03, issue 64d5f13)` banners in their module docstrings, covering the entire files; grep confirms no JSON-manifest reads and no non-legacy carve-outs inside the large file (I read its first 1000 lines to verify the banner scope — it is a frozen pre-contract materializer throughout). `_selected_eval_rollouts.py` (126 lines) is not LEGACY and was read in full.

### output_feedback_phase_modulated_recurrent.py

**Category A**
(none)

**Category B**
(none)

**Category D**
(none)

**Skipped (LEGACY)**
| file:lines | why skipped |
|---|---|
| output_feedback_phase_modulated_recurrent.py:1-3221 | Module-level `LEGACY (frozen 2026-07-03, issue 64d5f13)` banner in the module docstring (lines 1-8) covers the entire file. Pre-contract materializer; port-or-delete deferred to the report-stage era (feedbax 132f98c). No sub-block is exempted from the banner; grep confirms zero `json.load`/`read_text` manifest reads (the file only *writes* its own note/manifest/npz). |

### cs_stochastic_phase1.py

**Category A**
(none)

**Category B**
(none)

**Category D**
(none)

**Skipped (LEGACY)**
| file:lines | why skipped |
|---|---|
| cs_stochastic_phase1.py:1-787 | Module-level `LEGACY (frozen 2026-07-03, issue 64d5f13)` banner in the module docstring (lines 1-13) covers the entire file. Frozen Monte Carlo materializer; only durable I/O is *writing* its own note/manifest/npz (`write_outputs`, lines 483-528), no manifest reads. |

### _selected_eval_rollouts.py

**Category A**
(none) — no JSON/manifest file reads anywhere; the module converts in-memory JAX rollout states to host arrays.

**Category B**
(none)

**Category D**
| site | file:lines | what it does | sources read | classification | blocker/rationale |
|---|---|---|---|---|---|
| `initial_effector_position` | _selected_eval_rollouts.py:116-126 | Availability-probe cascade over `trial_specs.inits.values()`: `getattr(init_state, "pos", None)`, else shape-sniff (`shape[-1] &gt;= 2`) and slice `[..., 0:2]`, else raise | in-memory feedbax trial-spec PyTree (no file) | NOT eligible | Smells like hand-rolled `exists`/`has_type` predicate logic, but operates on live JAX PyTrees, not manifest/spec payloads bound in an `ExpressionContext`; the shape-sniff fallback also produces a computed slice, beyond pure extraction. Query language does not apply. |

**Skipped (LEGACY)**
(none)

**Batch summary:** zero migratable sites in this batch. Two of the three files are entirely LEGACY-frozen under issue `64d5f13` (module-level banners — the audit's skip rule applies to the whole files, so their internal manifest-*writing* code is not a migration target). The third file is an in-memory JAX optimization shim with no manifest/spec wrangling; its one borderline duck-typing cascade operates on live PyTrees outside the query language's domain.
