# A1 local engineering-smoke execution evidence

This packet is **non-scientific engineering-smoke evidence**. It does not answer
the cross-architecture certificate question.

## Outcome

The frozen six-row matrix now authors, lowers, enters governed storage, validates,
and has a portable hash-bound sidecar. No A1 row has trained. Execution is held
because independent preflight showed that the required evaluation and figure graph
cannot yet consume the resulting checkpoints and produce the frozen acceptance
packet.

This supersedes the earlier fresh-matrix launcher failure recorded in
`runs/execution_status.json`. That file remains historical structured evidence of
the first failed attempt; it is not the current readiness verdict. The launcher and
matrix portability defects were repaired in bounded lanes. The remaining block is
downstream and must not be bypassed by synthetic certificate rows or a private
figure path.

## Current identities

| Item | Identity |
|---|---|
| RLRMP feature head at consolidation | `9dc3c5dba81560635961dc3a309783e412b27e6e` |
| Protected/pinned Feedbax `develop` | `060d65d285969ec11e4a284712913550c462ba18` |
| Exact accepted-run Feedbax staging | `a86f6b8685d5ce6a2761d26a814b65528b9dee1a` |
| Current clean signed Feedbax staging | `257573ea7642b6570d12afac8a71ee913256e93a` |
| Implemented staging merges | `6e0352ab` ([issue:7e4cf6b]); `c2932138` ([issue:ca2f937]); `257573ea` ([issue:d81a868]) |
| Frozen matrix SHA-256 | `78108ca2286af701583e5c4eb87a92736820b5c9260129637722c61831a9e52f` |
| Portable sidecar URI | `repo://results/4eb51ee/runs/matrix.json` |
| Cross-lens intent SHA-256 | `7ada9db0fc412e9cd19b0e8a77308e7d295151c08cf05ee3fb0c54c02cbf62b6` |

[issue:238eaea] owns exact matrix reproducibility. [issue:e093cd9] owns portable
sidecar emission and materialization. Both are complete on this branch; neither
changed the frozen matrix bytes.

## Preflight evidence and blockers

Authoring/lowering passes for all six static-gain, augmented-linear, and GRU rows.
The downstream road is blocked as follows:

- `analysis/cross_lens/spec.json` is descriptive intent rather than an executable
  Feedbax `AnalysisBundleSpec`/`FigureSpec`.
- Its grouped stage sets `include_bundle_inputs=true`; the registered standard
  certificate analysis accepts only `EvaluationRunManifest` inputs.
- No configured evaluation recipe emits canonical cached
  `standard_certificate_rows` from real training/checkpoint lineage.
- Only the augmented-linear component provider is registered. Static-gain and
  empirical-nonlinear producers are absent, while the augmented provider still
  needs governed checkpoint and same-basis reference identities.
- No executable certificate-agreement figure template consumes the grouped analysis
  with reason-coded `not_applicable` handling.
- The registered bridge certificate report renderer is present and custody-routed,
  but has no real analysis manifest to consume.

Exact ownership:

- [issue:0d6c2ae] — augmented-state reference evidence;
- [issue:0be2b69] — real-manifest certificate-row producers; and
- [issue:6fa0431] — executable bundle and figure.

[issue:7e4cf6b], [issue:ca2f937], and [issue:d81a868] are implemented on current
staging and are not active blockers; [issue:d81a868] is `done`. Current worker
states are `in_progress` for [issue:0d6c2ae], and `blocked` for [issue:0be2b69]
and [issue:6fa0431].

## Stage verdicts

| Stage | Verdict |
|---|---|
| Tracked authoring and exact frozen bytes | pass |
| Governed emission and portable sidecar | pass |
| Six-row validation/lowering | pass |
| Local training | **blocked; not run** |
| Batch-50 checkpoint/resume | not generated |
| 24 evaluation manifests | not generated |
| Grouped certificate analysis | not generated |
| Figure manifest/render | not generated |
| Custody report render | not generated |
| Initial plausibility | not observed |
| Scientific evidence | none |

There are no A1 losses, movements, checkpoints, manifests, figures, or report
artifacts to interpret. `not_applicable` is reserved for structurally undefined
certificate components; it is not used for these missing pipeline products.

## Commands and evidence boundary

The public authoring path is:

```bash
FEEDBAX_STAGING="$HOME/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/integration__509368b-feedbax-staging"
PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python \
  scripts/emit_heterogeneous_training_matrix.py \
  --base-intent results/4eb51ee/runs/base.intent.json \
  --matrix-authoring results/4eb51ee/runs/matrix.authoring.json \
  --issue 4eb51ee \
  --output results/4eb51ee/runs/matrix.json

PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python scripts/launch_training.py validate \
  results/4eb51ee/runs/matrix.json

PYTHONPATH="$PWD/src:$FEEDBAX_STAGING" uv run --no-sync python scripts/launch_training.py dry-run \
  results/4eb51ee/runs/matrix.json
```

Cold public emission reproduced the exact matrix SHA and the portable sidecar
without hand-normalization. Accepted execution evidence used Feedbax staging
`a86f6b86`. The clean signed staging head is now `257573ea`; the protected pin
remains `060d65d`.

No execute command was released after the downstream preflight. No cloud, pod,
Modal, alternate executor, direct writer, synthetic checkpoint, injected
certificate row, extra seed, expanded batch budget, full suite, push, or protected
auth was used.

## KPI and bypass inventory

The revision-pinned A1 KPI record reports 302 authored production/spec LOC, one
generated matrix line, `c1=127`, and `c2=c3=c4=c5=0` at revision
`7f3b503c3f02c0efeea957aa3265ae8c6d1886eb`. These counts apply to A1 authoring;
the four bounded road issues remain separately attributable.

Bypass inventory is empty. The existence of blocked training does not increment
`c4`, because no escape hatch was invoked.

## Independent falsification checklist

1. Verify branch, protected pin, accepted-run staging, and current staging
   identities without conflating them.
2. Recompute the matrix and sidecar hashes from a different checkout.
3. Re-run cold public emission and require byte-identical matrix output.
4. Validate all six row IDs, seed 42, architecture, certificate mode, and training
   distribution; keep the four evaluation lenses out of the training axes.
5. Confirm the cross-lens intent fails executable bundle/figure validation for the
   documented reasons and that current evaluation recipes do not emit
   `standard_certificate_rows`.
6. Require the four owning issues to land before releasing training.
7. After release, require six training manifests, 24 evaluation manifests, one
   grouped analysis, one figure, and one custody report with matching identities.
8. Reject any hypothesis claim from this one-seed, 100-batch engineering smoke.
