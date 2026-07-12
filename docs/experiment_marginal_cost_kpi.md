# Experiment marginal-cost KPI

Record this KPI once per experiment issue at closeout. It preserves the frozen
`c1`–`c5` definitions established by issue `2c70801`:

- authored production LOC: tracked specs, configuration, and experiment-specific scripts;
- authored spec LOC: the spec/config subset of authored production LOC;
- c1: distinct authored JSON keys, plus explicitly listed concepts for formats the tool
  cannot parse semantically;
- c2: new registry entries;
- c3: callbacks authored for framework surfaces;
- c4: escape-hatch invocations, each of which should have a stated reason;
- c5: non-boilerplate control-flow constructs; and
- generated materialization LOC: tracked generated expanded payloads, reported separately
  and never counted as authored cost.

Create `results/<issue>/notes/marginal_cost_input.json` with this shape:

```json
{
  "experiment_issue": "abcdef0",
  "authored_production_paths": ["results/abcdef0/runs/matrix.json"],
  "authored_spec_paths": ["results/abcdef0/runs/matrix.json"],
  "generated_materialization_paths": [],
  "c1_extra_concepts": [],
  "concepts": {
    "c2_new_registry_entries": 0,
    "c3_authored_callbacks": 0,
    "c4_escape_hatch_invocations": 0,
    "c5_non_boilerplate_control_flow": 0
  }
}
```

After committing the experiment, record a revision-pinned report:

```bash
uv run --no-sync python scripts/experiment_kpi.py \
  results/<issue>/notes/marginal_cost_input.json --revision HEAD \
  --output results/<issue>/notes/marginal_cost.json
```

Review and commit the report, attach or register it with the experiment issue,
and include its headline counts in the verdict/closeout comment. Classifying the
paths and c2–c5 counts is a reviewable human assertion; the script makes their
line and JSON-key counts deterministic against immutable Git blobs.
