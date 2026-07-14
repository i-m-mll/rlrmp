# Graph sidecar fidelity audit (issue e9fc384)

## Purpose

This is a retroactive structural fidelity audit of every tracked archived
Feedbax graph sidecar (`model.graph.json`) under `results/`, done before issue
[ae15851] converts these historical sidecars into clean new-format
`GraphSpec` recipes and uses them as loadability regression fixtures. The risk
is asymmetric: it is cheap to inspect graph structure now, and expensive to
untangle a bad historical sidecar after it has been enshrined as a ground-truth
fixture. The audit classifies each sidecar by **inspected graph structure**
and **audited content hash**, not by production date or file path alone.

## Corpus

`git ls-files 'results/**/model.graph.json' 'results/**/*.graph.json'`
returned exactly **38** tracked files at audit time, matching the count expected
by the issue spec. All 38 were audited; no additional patterns were needed.
Issue `b6b5502` later retired the two known-wrong `30f2313` subjects with the
tagged legacy stock. Issue `dd7234e` subsequently added
`results/ef9c882/runs/base.graph.json`, a current-schema C&S graph whose
inspected `LinearStateSpace` structure matches its explicit `cs_lss` family.
The live guarded manifest therefore contains 37 clean sidecars: 36 surviving
historical point-mass sidecars plus that one native C&S-LSS sidecar. Converted
`ae15851` fixtures are excluded from this source-sidecar inventory so the
builder and drift guard audit the same corpus.

Every one of the 38 sidecars:

- carries `metadata.version == "rlrmp.feedbax_graph.v1"` (38/38), and
- has **no top-level `schema_version` key** (0/38) — `schema_version` only
  appears as a nested field inside sibling `run.json` / `model.graph.manifest.json`
  files (e.g. `"rlrmp.cs_stochastic_gru.v1"`), never inside `model.graph.json`
  itself.

Both facts were verified programmatically for all 38 files (not assumed) and
are recorded per-file in the manifest.

The later `ef9c882` sidecar intentionally differs from those historical
invariants: it carries a top-level current graph schema and no legacy
`metadata.version`. Its inspected node types include `LinearStateSpace`,
`StateFeedbackSelector`, `Subgraph`, and additive `Sum` adapters, so the audit
classifies it cleanly as native `cs_lss` rather than treating it as a legacy
point-mass conversion candidate.

## Method

For each sidecar, the audit script
(`results/e9fc384/scripts/build_graph_sidecar_audit_manifest.py`) parses the
JSON and inspects the `nodes` mapping (dict of node-id → `{type, params,
input_ports, output_ports}`). It checks five structural predicates per file:

1. `metadata_version_is_v1` — `metadata.version == "rlrmp.feedbax_graph.v1"`.
2. `no_top_level_schema_version` — no top-level `schema_version` key.
3. `point_mass_filter_channel_family_present` — the node-type set contains
   `{FirstOrderFilter, PointMass, RLRMPFeedbackChannels,
   RLRMPSimpleStagedNetwork}` (the point-mass/filter/channel family).
4. `linear_state_space_absent` — the node-type set does **not** contain
   `LinearStateSpace` (the CS-LSS family marker).
5. `controller_kind_recorded` — `nodes.net.params.controller_kind` is one of
   `{gru, vanilla_rnn}`.

The **actual** structural family is derived purely from the inspected node
types (`cs_lss` if `LinearStateSpace` is present, `point_mass` if the
point-mass/filter/channel set is present, `unrecognized` otherwise). The
**expected** conversion family is assigned per path by explicit human audit
(not a filename heuristic): all paths default to `point_mass` except the two
`results/30f2313/runs/cs_stochastic_gru__*/model.graph.json` sidecars, which
are audited as CS-LSS candidates (see below). A file is `clean` when expected
matches actual; otherwise it is `known_wrong`.

## Original result: 36 clean, 2 known-wrong

| Classification | Structural subfamily | Count | Representative paths |
|---|---|---:|---|
| clean | `point_mass_gru` | 32 | `results/b41c940/migrated/{2bc95fd,3702f54,b399efc,efc4d68,f47abb1}/*gru*/model.graph.json` |
| clean | `point_mass_vanilla_rnn` | 4 | `results/b41c940/migrated/efc4d68/baseline_vrnn__{jerk,none,smooth,smooth_jerk}/model.graph.json` |
| known-wrong | `point_mass_gru` (expected `cs_lss`) | 2 | `results/30f2313/runs/cs_stochastic_gru__{hidden_penalty,no_hidden_penalty}/model.graph.json` |

All 36 clean files share the node-type set `{Channel, FirstOrderFilter,
FixedField, PointMass, RLRMPFeedbackChannels, RLRMPSimpleStagedNetwork}` with
`net.params.controller_kind` recording either `gru` (32 files) or
`vanilla_rnn` (4 files, all under `results/b41c940/migrated/efc4d68/`). This is
the expected point-mass/filter/channel family for these historical
point-mass-plant baseline and ablation runs, so all 36 are classified `clean`.

By content hash, the 32 `gru` files collapse to one hash
(`7435864d...7aaff`, truncated) and the 4 `vanilla_rnn` files collapse to a
second hash (`1f31b996...97aaff`, truncated) — expected, since the graph
sidecar only encodes model/plant *structure*, and every run in a given
controller-kind family shares that structure even though hyperparameters
(loss weights, schedules, etc.) differ per run. Full hashes are in the
manifest.

### Known-wrong: `results/30f2313/runs/cs_stochastic_gru__*`

Both `cs_stochastic_gru__hidden_penalty` and
`cs_stochastic_gru__no_hidden_penalty` are **byte-for-byte identical**
`model.graph.json` sidecars (same sha256). Their node-type set is
`{Channel, FirstOrderFilter, FixedField, PointMass, RLRMPFeedbackChannels,
RLRMPPlantProcessForceNoise, RLRMPSimpleStagedNetwork}` — the point-mass
family (with one extra `RLRMPPlantProcessForceNoise` node for plant process
noise), and **no `LinearStateSpace` node**.

These two runs are the Crevecoeur/Cluff/Scott (C&S) 2019-oriented "CS
stochastic" nominal-fidelity study (issue `30f2313`). Their own `run.json`
sidecar (`results/30f2313/runs/cs_stochastic_gru__hidden_penalty/run.json`)
records an analytical C&S plant with a **48-dimensional delay-augmented
state** (`game_card.plant`: `physical_state_dim: 8`, `state_dim: 48`,
`disturbance_dim: 8`, `bw_contract: "top physical 8x8 block is identity; lag
rows are zero"`) — i.e. an analytical `LinearStateSpace`-with-delay
formulation, not a point mass. The same `run.json` is explicit that the
*executed* model did not use this analytical plant:

These historical nested recipes were retired under issue `ef8e1df`; recover them from git tag `legacy/ef8e1df-nested-run-json-retired` (the bytes are also in Mandible custody).

- `feedbax_graph.execution_backend: "rlrmp.legacy_simple_feedback_compat"`
- `fidelity_status.analytical_delay_augmented_state_input: false`
- `fidelity_status.exact_fidelity: false`
- `fidelity_status.certificate_lens: "input_output_map_certificate"` (not
  `"same_coordinate_gain_certificate"`)
- `model_structure.certificate_coordinate_claim: "not_same_coordinate_gain"`

So the run's own provenance corroborates the sidecar audit: this run was
trained through the legacy point-mass compatibility graph (`FirstOrderFilter →
PointMass`), not through the analytical CS-LSS plant its game-theoretic cost
card describes. The `model.graph.json` sidecar faithfully records what was
*executed* (point-mass), but that is the wrong structural family for a "CS-LSS"
loadability fixture — a future `ae15851` fixture consumer expecting
`LinearStateSpace` + delayed feedback for this issue would get a silently
wrong plant. Both files are therefore classified **known_wrong** with
`expected_conversion_family: cs_lss`, `structural_family_actual: point_mass`,
and are excluded from `ae15851`'s fixture set pending a corrected/annotated
CS-LSS sidecar (out of scope for this audit — see Non-Goals).

## Structural predicates: full pass/fail

All five predicates pass for all 38 files **except**
`point_mass_filter_channel_family_present`/`linear_state_space_absent` are
still both `true` for the two known-wrong files (they *are* point-mass, and
correctly lack `LinearStateSpace` — the mismatch is that they are *expected*
to be `cs_lss` and are not). No file fails `metadata_version_is_v1`,
`no_top_level_schema_version`, or `controller_kind_recorded`. There were no
surprises relative to the issue spec's stated evidence: the 30f2313 pair
matched the spec's "confirmed live" point-mass-shape finding exactly, and the
4 `baseline_vrnn__*` sidecars matched the spec's stated `vanilla_rnn` family
exactly.

## Manifest

The machine-readable manifest is
`results/e9fc384/notes/graph_sidecar_audit_manifest.json`
(`schema_version: 1`). Per file it records: `path`, `sha256`,
`metadata_version`, `top_level_keys`, `has_top_level_schema_version`,
`retired_component_ids` (the legacy graph's node-id keys — e.g. `efferent`,
`feedback`, `force_filter`, `mechanics`, `net`, `plant_intervenor`, and, for
the CS-stochastic pair, `plant_process_force_noise`), `node_types`,
`structural_types`, `controller_kind`, `structural_family_actual`,
`structural_subfamily_actual`, `expected_conversion_family`,
`classification`, `classification_reason`, `structural_predicates_checked`,
and a `conversion_candidate_key` object
(`retired_type`, `structural_predicate`, `metadata_version`,
`audited_fixture_hash`) matching the composite key the issue spec requires.
The current `ef9c882` entry is likewise hash-pinned and classified from its
node types while retaining its native top-level schema fields in the recorded
inspection data.

`tests/test_graph_sidecar_audit_manifest.py` guards this manifest against
drift: it fails on an empty manifest, a live `git ls-files` count that
disagrees with the manifest's `audited_count`, any live sidecar path missing
from the manifest (or vice versa), any sha256 mismatch between the manifest
and the live file, and any tracked sidecar not present in the manifest at all
(the "unaudited new sidecar" case the issue spec's acceptance criteria
require).

## Non-goals (explicitly out of scope for this audit)

- This audit does not convert any sidecar to the new `GraphSpec` format.
- This audit does not define a generic historical `GraphSpec` translator.
- This audit does not treat successful materialization/loadability as
  sufficient evidence of fidelity — the CS-LSS pair would very plausibly
  *load* successfully (it's a well-formed point-mass graph); it is still
  structurally wrong for its intended purpose.
- Correcting or replacing the two known-wrong sidecars with a faithful
  CS-LSS graph is left to a follow-up (tracked separately; not this issue).
