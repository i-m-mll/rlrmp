# Destination-based data-in-code gate: strengthened contract

This is the committed normative specification for the `caa3f1b` gate after the
post-integration verification in `_artifacts/31aaa31/verification/gates.md`. It supersedes
the coverage-target and D1-rollout portions of the original design artifact at
`_artifacts/31aaa31/design/data_in_code_gate.md`; the original data-in-code criterion,
five-detector architecture, scan roots, and D5 delegation remain unchanged.

## Enforcement contract

The gate scans Python under `src/`, `scripts/`, and `results/*/scripts/` and reports five
destination-oriented shapes:

1. `argv_rows`: CLI rows containing a flag and a numeric literal.
2. `spec_flow`: numeric literals passed directly into known governed-spec constructors.
3. `default_bundle`: static parameter bundles returned by named builder functions or
   scattered through config/parameter class defaults.
4. `hp_constant`: literal-backed module or uppercase class constants whose names identify
   a run, evaluation, or analysis parameter.
5. `empirical_table`: the existing generated/empirical numeric-table detector.

`argv_rows` in `src` is enforced: every live finding must have a curated, documented
allowlist entry. An enforced finding is never baseline-eligible. Ratchet findings may be
committed in `ci/data_in_code_baseline.json`; advisory findings are reported but are not
written to that file. This distinction is mechanically checked by tests.

The sole D1 `src` exception is `smoke_train_command` in
`closed_loop_distillation.py`. Its literals are fixed one-batch smoke safety bounds; the
governed run spec is supplied by the caller. This is a permanent, rationale-bearing
exception rather than an anonymous baseline entry.

## Static replay eligibility and targets

Replay reports must publish both the raw historical denominator and the independently
defined static-eligibility denominator. Passing only the eligible rate is not sufficient
reporting.

| Detector/class | Eligible population | Acceptance target |
|---|---|---:|
| D1 `argv_rows` in `src` | Live AST shapes, after sibling migrations | Zero non-allowlisted findings |
| D2 `spec_flow` | Direct numeric literal keyword arguments at known spec-constructor calls | Positive/negative canaries and live reporting; no historical percentage floor |
| D3 `default_bundle` | Inventory bundle objects with a statically owned literal bundle or class defaults | At least 95% caught |
| D4 `hp_constant` | Inventory constant objects whose exact assignment still exists, has a numeric literal tree, and is not a dimension, tolerance, schema, path, label, or other named exemption | At least 95% caught |

The former aggregate target of at least 60% for historical
`run_hyperparameters` rows is revised to a **0% acceptance floor, report-only** for D2/raw
historical coverage. This is a reduction in the formal target, not a claim that low recall
is desirable. The inventory mixes declarations that D2 cannot observe with direct spec
calls, and includes bare scalars whose intent cannot be classified reliably from syntax.
For example, a neutral `D = 1.0` may be a scientific target, a display scale, a numerical
guard, or a dimension. Treating every such scalar as a run parameter would replace misses
with an unbounded false-positive channel.

D2 therefore keeps a high-precision contract: direct literal keywords at known spec sinks
must be caught, indirect values must not be guessed, and the live count is always reported.
D4 carries the measurable replay target for statically named literal constants. The raw
all-row rate remains evidence and must not be presented as the eligible D4 rate.

## Detector details added by the strengthening pass

D3 follows returned locals, nested literal dictionaries, `Namespace`-style expansion,
and incremental literal `dict.update(...)` calls. Builder-name gating and minimum bundle
cardinality keep result/summary dictionaries out of the finding set. Config/parameter
classes with at least two static parameter defaults are covered; schema/runtime models and
purely dynamic or default-free models are excluded.

D4 accepts nested literal containers, simple literal expressions, NumPy/JAX array wrappers,
plural and numbered name tokens, and the irregular plural `RADII`. Its reviewed lexicon
includes compound names observed in the inventory. It rejects dimension, tolerance,
schema, path, label, and string-only constants. Lowercase class fields remain D3's domain,
which avoids duplicate D3/D4 findings.

The tree walk fails closed when a scanned Python file cannot be parsed. A syntax error may
not turn a source file into an unreported scanner blind spot.

## Required verification

- Run `tests/test_data_in_code_scan.py` through `scripts/dev_tests.sh`.
- Replay against `_artifacts/b6136cc/second_order/data_in_code_inventory.jsonl` and commit
  the raw and eligible rates plus exclusions.
- Report the live finding matrix, allowlist count, ratchet baseline size, and violations.
- Do not add a baseline key or allowlist exception merely to make a new finding green.
  Baseline regeneration is appropriate only when detector strengthening reclassifies
  already-live ratchet findings, and the resulting shape must be documented.

Co-Authored-By: Codex (GPT-5) <codex@openai.com>
