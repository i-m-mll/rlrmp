# Stage-2 Feedbax CLI launch source

`stage2_cli_rows_source.json` is the canonical three-row input for rebuilding
the Stage-2 deployment manifest. It names the governed outer RLRMP run specs;
the builder validates and extracts each embedded `TrainingRunSpec`, writes the
three CLI-ready nested specs, and emits commands that enter through
`python -m feedbax execute-training-run-spec --resume`.

After the Stage-2 run-spec files are present on the owning deployment branch,
regenerate the manifest from the repository root with:

```bash
PYTHONPATH=src uv run --no-sync python scripts/build_feedbax_cli_rows_manifest.py \
  results/c6c5997/deploy/stage2_cli_rows_source.json \
  results/c6c5997/deploy/stage2_rows_manifest.json
```

The generated `feedbax_training_run_specs/` directory and rows manifest form
one launch packet. Do not hand-edit the generated commands or replace them with
private RLRMP launcher imports.
