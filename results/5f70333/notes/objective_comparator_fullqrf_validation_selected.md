# Full-QRF objective comparator sidecar

Scope: validation-selected checkpoints for the two full-QRF C&S GRU rows.

Placement recommendation: keep this as an optional objective-comparator sidecar adjacent to evaluation diagnostics, not as a standard-certificate gate. It depends on a named analytical comparator and implemented validation scalar.

Caveat: GRU values are validation-selected training-validation objectives under `full_analytical_qrf`; the extLQG value is the reference sidecar `extlqg_expected_cost` from the local `extLQG -> computeOFC -> computeExtKalman` fixed-point comparator. Ratios are objective-lens diagnostics only.

| run | mean selected validation | mean best logged validation | extLQG expected cost | selected/extLQG | best/extLQG |
|---|---:|---:|---:|---:|---:|
| `lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64` | 4367.3 | 4357.59 | 12201.4 | 0.357934 | 0.357138 |
| `lss_stabilization_fullqrf_warmcos__lr3e-3_clip5_b64` | 4342.64 | 4335.33 | 12201.4 | 0.355912 | 0.355313 |

JSON sidecar: `results/5f70333/notes/objective_comparator_fullqrf_validation_selected.json`
