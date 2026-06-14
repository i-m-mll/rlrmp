# Package Layout

Issue `5a308f3` made the remaining package-boundary decision after the lower-risk
cleanup lanes moved analysis shims, benchmark/cloud code, controllers, and model
construction out of ambiguous locations.

## Canonical Map

| Surface | Canonical package | Rationale |
| --- | --- | --- |
| Training methods and task/model construction | `rlrmp.train` | Stable training-method package already used by drivers, reload paths, and Feedbax plugin registration. |
| Durable run-spec validation | `rlrmp.runtime.run_specs` | Runtime contract for tracked training recipes and post-run ingestion, not a training algorithm. |
| RLRMP structured-spec schema policy | `rlrmp.runtime.spec_migrations` | Durable manifest/schema registration used by plugin startup, run specs, and analysis sidecars. Schema identities are preserved. |
| Feedbax Studio materialization | `rlrmp.runtime.studio_records` | Service/runtime bridge from completed TrainingRunManifest records to Studio workspaces and manifests. |
| Mirror-tree path helpers | `rlrmp.paths` | Cross-cutting public utility used by training, analysis, scripts, and cloud code. Moving it would add churn without clarifying ownership. |
| Auto-generated Markdown section I/O | `rlrmp.io` | Cross-cutting file utility with documented use in analysis scripts; the root name is concise and not experiment-specific. |

Durable schema identifiers such as `rlrmp.run_spec` and
`rlrmp.gru_evaluation_diagnostics` stay unchanged. Only the owning Python module
metadata moved from the previous root schema-policy module to
`rlrmp.runtime.spec_migrations`, and new Studio-generated metadata now reports
`rlrmp.runtime.studio_records`.
