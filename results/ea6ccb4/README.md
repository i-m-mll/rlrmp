# ea6ccb4 — Generated calibration/budget data moved out of source code

Generated and adopted empirical data that previously lived as module-level Python
constants now lives here as governed, schema-versioned data products carried on
the Feedbax `AnalysisDataProduct` envelope and loaded at runtime by typed product
identity with fail-closed validation (`rlrmp.data_products`). Source code keeps
schemas, loaders, and builders; it does not embed generated datasets as source
constants (see the policy in `AGENTS.md`/`CLAUDE.md`).

## Data products

- `data_products/perturbation_open_loop_calibration.json`
  - `product_schema_id`: `rlrmp.perturbation_open_loop_calibration`
  - `product_schema_version`: `rlrmp.perturbation_open_loop_calibration.v2`
  - `role`: `perturbation_open_loop_calibration`
  - `product_identity_hash`: `03edd3141b62d1b1cf045097114caac7bc96f1236a433875976aec974d9bb97a`
  - Holds the open-loop unit-sensitivity table (`peak delta x` per family/timing bin)
    and the controller-visible native velocity scale (the C&S faithful-plant LQR peak
    forward velocity, adopted from `a7dad8a`). Replaces the deleted constants
    `DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT` and
    `DEFAULT_CONTROLLER_VISIBLE_VELOCITY_SCALE_M_S`.
  - Regeneration: `uv run python scripts/materialize_perturbation_open_loop_calibration.py`
    (extLQG nominal-command open-loop replay). The distilled table here is the runtime
    product; the bulk per-row manifest stays under `_artifacts/1ad3c16/`.
  - Loader: `rlrmp.data_products.calibration.load_open_loop_calibration`.

- `data_products/broad_epsilon_budget_anchors.json`
  - `product_schema_id`: `rlrmp.broad_epsilon_budget_anchors`
  - `product_schema_version`: `rlrmp.broad_epsilon_budget_anchors.v1`
  - `role`: `broad_epsilon_budget_anchors`
  - `product_identity_hash`: `4e5d319c4848ef19d25ddf9dc8d21a6230cc0d336c5f565fe1a0b63516332542`
  - Per-level closed-loop epsilon budgets adopted from analytical sources, with an
    explicit adoption record per level: `moderate` (gamma factor 1.4) from
    `results/cb98e58/notes/analytical_game_card_manifest.json` and `strong` (gamma
    factor 1.05) from `results/a7dad8a/notes/adversary_equivalence_manifest.json`.
    Replaces the deleted `BROAD_EPSILON_LEVELS` constant.
  - Regeneration: `rlrmp.data_products.broad_epsilon.build_broad_epsilon_budget_anchors_product`
    reads the analytical frontier entries directly.
  - Loader: `rlrmp.data_products.broad_epsilon.load_broad_epsilon_anchors`. On load it
    re-reads the analytical sources and fails closed if the persisted values ever drift
    from them, so historical runs stay reproducible through the adoption records.

## Reproducibility

Both products carry their identity hash and provenance. The broad-epsilon product's
per-level `adoption` record notes that historical broad-epsilon runs used the
identical baked values; the calibration product records the materializer and the
adopted-scale provenance. The AST data-lint
(`rlrmp.data_products.lint`, enrolled as `generated_data_constant_scan` in
`ci/feedbax-contract-suite.toml`) prevents regenerated data from re-accreting as a
source constant.
