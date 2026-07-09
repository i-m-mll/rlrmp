# CI gate proposals: code-archaeology audit (issue `05883e7`)

Per CLAUDE.md's "a fix without a guard is half a fix" principle: this note
proposes one gate family per violation class the audit found, in the same
style as the existing `feedbax_contract` families documented in the
worktree's `CLAUDE.md` (`ci/feedbax-contract-suite.toml`). Filing/implementing
any of these is itself remediation work and is out of scope for this audit —
each adopted gate graduates to its own issue, same as the portfolio items in
`results/05883e7/notes/remediation_portfolio.md`.

Every proposal below is assessed for false-positive risk, because this audit
surfaced two concrete mechanisms that make naive static gates unsafe here:
**registry-wired liveness** (an object reached only through
`component_registry`/recipe-registry string dispatch shows zero routed
inbound references even though it's genuinely live) and **cross-language
liveness** (Feedbax Studio's TypeScript frontend references Python object
names as data; Python-side static analysis cannot see this — see
`results/05883e7/notes/findings_digest.md` Section 4). Any gate that flags
"unused" or "dead" code must exclude registered objects, string-hit objects,
and (where relevant) grep the TypeScript `feedbax/web/src/` tree, or it will
produce false positives exactly where the audit already found them.

## 1. `experiment_named_in_src` scan

**What it scans:** AST + name-pattern check over `src/rlrmp/` for module- or
object-level identifiers that encode an issue-hash or experiment-specific
token — e.g. the `planned_<hash>_*_rows` naming convention
(`planned_020a65b_h0_pgd_rows`, `planned_e4800d6_sisu_spectrum_rows` in
`cs_perturbation_training.py`) or locked run-row functions named after a
completed experiment's tracking issue (`cs_nominal_gru.py`'s rows for
`ef9c882`). Mechanically: a regex over `src/rlrmp/**/*.py` object and module
names for a 7-character hex-like token, optionally combined with a
`planned_`/`_rows`/`row_` naming shape, cross-checked against whether the
matched token resolves to a real Mandible issue id (a false-positive filter:
not every 7-hex-char substring is an issue reference).

**What it protects:** The script-placement policy's rule that
experiment-specific code lives in `results/<hash>/scripts/`, not baked into
generically-named `src/rlrmp/` modules as named literals. This is precisely
the audit's motivating symptom (`cs_nominal_gru.py`, `cs_perturbation_training.py`).

**Allowlist policy:** Shrink-only, matching `reaccretion_ratchet`'s existing
pattern. New `src/rlrmp/` code may not add new experiment-named objects
beyond the frozen baseline; the baseline itself only shrinks as class (e)
portfolio items are remediated.

**Precedent finding:** 110 modules / 362 objects already carry this flag
corpus-wide (class (e) of the remediation portfolio, 11,889 flagged LOC), with
`cs_perturbation_training.py` (9 objects, 874 LOC) and `cs_nominal_gru.py` (5-7
objects across different chunks, up to 795 LOC) as the largest concentrations.

**False-positive risk: low.** This is a naming-pattern check, not a liveness
check — it does not depend on xref/registry evidence, so the registry-wired
and cross-language caveats above don't apply. The main risk is over-matching
generic 7-character tokens that happen to look hash-like but aren't issue
references; mitigate by requiring the matched token to resolve against the
Mandible issue namespace (or a locally cached issue-id list) before flagging.

## 2. `hp_literal_ratchet` extension (extends `generated_data_constant_scan`)

**What it scans:** The existing `generated_data_constant_scan` AST lint
(`rlrmp.data_products.lint`) flags multi-entry high-precision float container
literals under `src/`. This extension adds two shapes it currently misses:
(a) functions whose entire body constructs and returns a dict/list literal
keyed by hyperparameter-like names (`lr`, `batch_size`, `seed`,
`gradient_clip_norm`, ...) — the `planned_*_rows`/`_base_hps`-style bundle
pattern; (b) whole locked-run-row-list functions, regardless of whether the
literal values themselves are floats or a mix of types (the current lint's
float-container trigger misses these when the payload is mixed-type argv
strings, as in `cs_perturbation_training.py`'s locked rows).

**What it protects:** The "data stays separate from code" non-negotiable
principle — the same one the base `generated_data_constant_scan` gate
protects, extended to catch the adjacent shape the base scan's float-only
trigger currently misses.

**Allowlist policy:** Shrink-only, following the base gate's own convention
("allowlists are shrink-only unless a new issue documents why an exception is
still required").

**Precedent finding:** 1,316 objects corpus-wide carry at least one
hp-like-signal flag in the census (a dict key/kwarg name that looks like a
hyperparameter, or a numeric-literal container of 4+ entries); the
`data_in_code` contract flag specifically appears heavily in
`cs_perturbation_training.py` (11 hits), `output_feedback_phase_modulated_recurrent.py`
(14 hits), and `cs_nominal_gru.py` (5 hits, 793 LOC) — none of which are
caught by the current float-only-container trigger, since these are the
mixed-type "locked row" and "planned rows" shapes.

**False-positive risk: medium.** Distinguishing "a function that returns a
literal hyperparameter bundle that should be a governed data product" from
"a function that legitimately constructs a small, bounded config dict
in-line" needs a size/shape threshold (the base gate's own "small
conventional constants... are out of scope" carve-out should carry over
here) and should stay allowlist-with-rationale for single small bundles, per
the base gate's existing exception mechanism.

## 3. `dangling_import_gate`

**What it scans:** Unresolvable repo-shaped imports — an import statement
whose target module resolves within the repo/dependency graph but whose
imported *name* is not actually exported from that module. Mechanically:
the same one-hop re-export-aware import resolution the Phase 3 dangling-
reference sweep already implements (`sweeps/dangling/`), run as a CI check
rather than a one-off audit pass, restricted to `dangling_import_name`
findings at `medium`/`high` confidence (imports inside `try: ... except
ImportError:` blocks are already downgraded to `low` by the sweep and should
stay excluded from a hard gate).

**What it protects:** Import correctness at CI time, rather than discovering
a broken import via a one-off archaeology pass years after it was introduced.

**Allowlist policy:** Not applicable in the shrink-only sense — this should
be a hard fail-closed gate once initial findings are triaged and fixed
(unlike the ratchet-style gates below, a dangling import is unambiguously a
bug once confirmed, not a judgment call that needs a grandfathered
allowlist). A short-lived allowlist during initial rollout is acceptable
while the 55 existing findings are triaged, but should burn down to zero
rather than persist as a permanent exception list.

**Precedent finding:** `feedbax/analysis/support.py:122` does
`from feedbax.intervene import AbstractIntervenor`, but `AbstractIntervenor`
is not exported from the `feedbax.intervene` package facade — a real,
confirmed bug (see `results/05883e7/notes/feedbax_backlog.md`). 55
`dangling_import_name` findings exist corpus-wide; most need triage before
this can be a hard gate (see false-positive risk below).

**False-positive risk: medium-high, requiring triage before hard-gating.**
The sweep's own documented limitation: this check chases one hop of
re-export but a name reachable only through a dynamic `__getattr__` or
`importlib`-based mechanism will still show up as a false positive — the
sweep already downgrades these to `low` confidence when the owning module
defines `__getattr__`, and a CI gate should honor the same downgrade rather
than hard-failing on every `dangling_import_name` hit indiscriminately.

## 4. `duplication_ratchet` (scoped to `src/rlrmp/analysis/pipelines/`)

**What it scans:** The same LSH/MinHash near-duplicate detector the Phase 3
duplication sweep already implements (bottom-k shingle sketches over
tokenized function/class/constant bodies), run in CI and compared against a
committed redundant-LOC budget for `src/rlrmp/analysis/pipelines/`
specifically — the hot zone the audit identified, not a repo-wide gate.

**What it protects:** Against new duplication accreting in the one directory
already carrying the most in-`src/` redundant LOC, without trying to
retroactively enforce zero duplication repo-wide (which the sweep's own
heuristic-limitation notes make clear would be too noisy to hard-gate
everywhere — framework-family clusters like feedbax's Equinox norm-layer
wrappers are legitimate, not defects).

**Allowlist policy:** Shrink-only redundant-LOC budget. New code may not
increase the pipelines-tree redundant-LOC total above the frozen baseline;
the baseline shrinks as class (c) portfolio items are remediated (e.g.
`output_feedback_phase_modulated_recurrent.py`'s in-file `_*disturbance*_response_maps`
duplication, dup_0011 in the portfolio).

**Precedent finding:** 115 clusters with at least one member under
`src/rlrmp/analysis/pipelines/`, contributing 5,017 redundant LOC — the
second-largest hot zone the sweep found (after `results/*/scripts/`, which
is explicitly out of scope for a `src/`-only ratchet since that tree's
duplication is a different, already-tracked problem — see remediation
portfolio class (c) and (d)).

**False-positive risk: low-to-medium.** The sweep's own documented
limitations apply directly: LSH bucket comparison can miss true near-dup
pairs whose sketches don't collide (under-counts, not over-counts, so safe
for a ratchet direction); literal-value differences (two hyperparameter
dicts with the same keys/shape but different values) score lower than a pure
structural diff would, which similarly biases toward under- rather than
over-flagging. Net: the heuristic is more likely to miss real duplication
than to falsely flag legitimate variation, which is the safe failure
direction for a shrink-only ratchet.

## 5. `dead_code_ratchet`

**What it scans:** Objects with zero routed inbound references (import- or
call-resolved), zero string hits, not `registered`, and not `__all__`-listed
— the same "strong dead candidate" signal definition `notes/audit_plan.md`
already establishes for Phase 0's xref output — tracked as a shrink-only
allowlist of currently-known dead objects, so *new* dead code cannot silently
accrete even though the *existing* backlog isn't force-deleted by the gate.

**What it protects:** Against the class of zero-consumer code this audit
spent most of its effort finding (class (a)'s 5,401 exhaustively-verified
LOC, plus the 602-object `needs_decision` backlog) recurring after cleanup —
mirrors the existing `reaccretion_ratchet` pattern for exactly this purpose.

**Allowlist policy:** Shrink-only. The initial allowlist is seeded from the
audit's own confirmed-dead corpus (class (a)); as those items are deleted
per the remediation portfolio, the allowlist shrinks and CI enforces that no
*new* object joins the zero-inbound/unregistered/no-string-hits population
without either being used or being explicitly reviewed and added to the
allowlist with a rationale (mirroring `generated_data_constant_scan`'s
allowlist-with-rationale exception path).

**False-positive risk: high without the registry/cross-language exclusions
— this is the gate most likely to misfire if built naively.** This audit
found the false-positive mechanism directly, twice: (1) the spark-vs-Sonnet
comparison (`spark_comparison/report.md`) showed a model applying a literal
"zero routed inbound references" rule to intra-module private helpers
produced 8 wrong `dead` labels out of 10 spot-checked, on code whose callers
were visible in the very file being scanned — **a gate must count
intra-module references, not just cross-module ones**, or it will flag
things like `induced_gain.py`'s `_W_CHANNELS`/`_validate_w_channel`/etc.
family (confirmed live via intra-module use in the classification
corpus-notes fix-up). (2) Three of Phase 4's 21 refutations trace to exactly
this gate's naive failure mode: `feedbax.plot.colors.adjust_color_brightness`
(called via a re-exported package namespace, `fbp.adjust_color_brightness(...)`),
`feedbax.tasks.task.safe_state_set` (public-export-and-test evidence missed
by routed-import resolution), and `feedbax.analysis.aligned.get_trivial_reach_directions`
(referenced by name from Feedbax Studio's TypeScript frontend data, invisible
to any Python-only static pass). **A `dead_code_ratchet` must exclude
`registered=true` objects, any object with nonzero `string_hits`, and should
additionally grep `feedbax/web/src/` for the object's bare name before
flagging anything under `feedbax/analysis/` or other Studio-reachable
surfaces** — otherwise it will reproduce the exact false-positive pattern
this audit had to spend a whole verification phase catching.

## 6. Module-size/purpose-entropy advisory (warning, not a hard gate)

**What it scans:** The same SPLIT-candidate score the Phase 5A synthesis
already computes deterministically: module LOC × purpose-entropy (bits,
across the corpus's 17-category purpose scheme) × (1 + distinct contract
flags present). Run in CI as an advisory check against a threshold (e.g.
flag any module scoring above some multiple of the current median), not a
hard failure.

**What it protects:** Against a module drifting toward the
`cs_nominal_gru.py` shape (8,902 LOC, 2.27 bits of purpose entropy, 5
distinct contract flags, SPLIT score 121,088 — nearly double the runner-up)
without anyone noticing until it's an ~8,900-line module mixing
spec-construction, training-loop, and hp-constant code.

**Allowlist policy:** Not applicable in the ratchet sense — this is
explicitly advisory, not a hard gate, precisely because of the confidence
caveat below. A manually-curated watchlist (the module_structure.md SPLIT
top-15) is the safer artifact; CI can warn when a module newly enters that
watchlist's score range, prompting a human look rather than blocking a
merge.

**Precedent finding:** `results/05883e7/notes/module_structure.md`'s SPLIT
ranking — `cs_nominal_gru.py` at 121,088, `cs_perturbation_training.py` at
63,462, five `analysis/pipelines/*` modules following the same pattern
(large module mixing a dominant purpose with hp/spec-construction and
multiple misplacement-shaped contract flags).

**False-positive risk: this is exactly why it must stay advisory, not a
gate.** The `purpose` field driving the entropy calculation is LLM-derived
classification, and per the QA calibration in `verification/calibration.md`
it has only 74.1% raw agreement (Cohen's kappa 0.706) between independent
reviewers — noticeably less reliable than `disposition` (97.0%) or
`generality` (98.6%). A hard gate on a score built from a field with
meaningfully lower inter-rater agreement would fail closed on noise as often
as on real signal; a warning that prompts human review is the appropriate
strength of enforcement for this particular signal, at least until purpose
classification itself is re-validated at higher reliability.

## Summary table

| Gate | Enforcement | False-positive risk | Precedent scale |
|---|---|---|---|
| `experiment_named_in_src` scan | shrink-only ratchet | low | 110 modules / 362 objects |
| `hp_literal_ratchet` extension | shrink-only ratchet | medium | 1,316 hp-signal objects corpus-wide |
| `dangling_import_gate` | hard fail-closed after triage | medium-high (needs `__getattr__`-aware downgrade) | 55 findings, 1 confirmed real bug |
| `duplication_ratchet` (pipelines-scoped) | shrink-only budget | low-medium | 115 clusters / 5,017 redundant LOC |
| `dead_code_ratchet` | shrink-only allowlist | **high without registry/cross-language exclusions** | 2,947 zero-inbound candidates corpus-wide; 5,401 LOC exhaustively confirmed |
| Module-size/purpose-entropy | advisory warning only | high (classification-noise-dependent) | `cs_nominal_gru.py` at ~2x runner-up |
