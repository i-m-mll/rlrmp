# C&S GRU Optimizer Stabilization Screen

Issue: `3e66604`

Rows:

| run | lr | clip | batch | replicates | backend | status |
|---|---:|---:|---:|---:|---|---|
| `lss_stabilization__lr3e-3_clip1_b250` | `3e-3` | `1` | `250` | `5` | `cs_lss` | completed |
| `lss_stabilization__lr3e-3_clip5_b250` | `3e-3` | `5` | `250` | `5` | `cs_lss` | completed |
| `lss_stabilization__lr1e-3_clip1_b250` | `1e-3` | `1` | `250` | `5` | `cs_lss` | completed |
| `lss_stabilization__lr1e-3_clip5_b250` | `1e-3` | `5` | `250` | `5` | `cs_lss` | completed |

All rows use AdamW with delayed cosine schedule, weight decay `0`, `nn_hidden=0`,
`n_train_batches=12000`, checkpoint interval `500`, and Modal A10 GPU execution.
The synced run specs intentionally set `feedbax_graph.graph_spec_path` to `null`
because the current compatibility GraphSpec exporter would otherwise serialize the
legacy `FirstOrderFilter -> PointMass` path for `cs_lss` runs.
