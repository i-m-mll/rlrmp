# LinearDynamicsAdversary: flavor-(b) ΔA model-class adversary

**Tracking issue:** `c723082` (closed). **Status:** experiment complete; results
synthesised. **Phase:** methodology-fix (`b557d4e`).

## Scope

End-to-end training + analysis runs for the flavor-(b) `LinearDynamicsAdversary`
(ΔA · x model-class disturbance) introduced as the formal counterpart to the
flavor-(a) additive force adversary. This dir bundles:

- `runs/flavor_b_eta{0.03,0.10,0.30}__seed_{0,1,2}.json` — 9 training-run specs
  across (η × seed).
- `runs/induced_gain_first_run.json`, `runs/induced_gain_flavor_b.json`,
  `runs/peak_velocity_flavor_b.json` — downstream analysis run-specs.
- `flavor_b_README.md` — narrative summary of the sweep.
- `notes/induced_gain_first_run.md`, `notes/induced_gain_flavor_b.md` —
  per-run analysis notes.

Historical nested `run.json` recipes were retired under issue `ef8e1df`; recover them from git tag `legacy/ef8e1df-nested-run-json-retired` (the bytes are also in Mandible custody).
- `notes/induced_gain_cross_method_comparison.md`,
  `notes/peak_velocity_cross_method_comparison.md` — cross-method comparisons.

## Cross-refs

- `72fb8d9` — Flavor-A vs flavor-B synthesis document (external-review).
- `b557d4e` — methodology-fix phase umbrella.
- `97c227a` — Riccati flavor-(b) extension (closed).
- `eb7fb9f` — caveat: "add LEQG aggregation" is not a flavor-(b) backdoor.

## Bulk artifacts

Live under `_artifacts/c723082/` (gitignored). Model checkpoints / training logs
were originally written to `_artifacts/part2_5/runpod/flavor_b/` and migrated to
the canonical hash-dir layout during the f485c26 reorg.
