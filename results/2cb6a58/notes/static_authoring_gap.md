# Static authoring gap: matrix rows cannot re-lower science axes

> Historical packet, superseded on 2026-07-13. The governed per-row re-lowering
> route is integrated and the exact four-row matrix now validates and emits. Current
> status is recorded in `engineering_smoke_evidence.md`: local execution is blocked
> before batch 1 by an unconditional source-checkpoint requirement for fresh
> matrices. This file remains as the pre-integration rationale for refusing a
> compiled-field patch forest.

Date: 2026-07-13

## Verdict

The selected force-observability family is blocked before governed matrix
authoring. The current matrix compiler applies row patches to an already-compiled
Feedbax `TrainingRunSpec` and validates the result; it does not apply a row to compact
RLRMP authoring intent and then run the registered science lowerers. Both selected
axes fan out across derived contract fields, so a small row override cannot express
the family faithfully.

No workaround was attempted. The escape-hatch inventory is empty, and the frozen
KPI classification remains `c2=c3=c4=c5=0`. No project import, matrix validation,
emission, custody write, training, test, or full-suite command ran.

## Static identity snapshot

| Item | Identity at inspection |
|---|---|
| RLRMP branch HEAD | `bd5292565ea56148734384ef8ee3393dce73832b` |
| pinned Feedbax `develop` | `f2377e659a70bb67b8a03a406dc49024b47bcef5` |
| pending ordered-lowerer Feedbax commit | `fc819ff0` via auth `d63e7780` |
| intended matrix | `results/2cb6a58/runs/matrix.json`; `blocked_not_generated` |
| four planned run IDs | `blocked_not_generated` pending governed compilation |

Relevant existing Git blob identities, recorded only as inventory rather than
adopted bases, are `ef65f10a528559361d431c9a04f6589aafa43039` for the
`cb3685a` nominal recipe, `5e370f882bfd3e3e3e39281c31d8216294959581` for the
`1ab1fef` broad-PGD recipe, and `64e6f2e3cc74f7691326d7a84b0e6053b074f097`
for the first `c6c5997` adaptive-epsilon recipe. These are Git blob IDs, not the
canonical-json base hash required by a future matrix.

## Exact compiler evidence

- `src/rlrmp/train/launch.py:86-90` says the launch surface accepts only a governed
  `TrainingRunMatrixSpec` and rejects flat family configs or nested run specs.
- `src/rlrmp/train/launch.py:122-128` hands that matrix directly to Feedbax
  `materialize_run_matrix`.
- At Feedbax commit `fc819ff0`, `feedbax/training/run_matrix.py:702-713` resolves a
  base payload, applies each row's override patches, and immediately calls the row
  validator. There is no RLRMP authoring or science-lowering callback between the
  patch and validation.

The current RLRMP dependency pin is still
`f2377e659a70bb67b8a03a406dc49024b47bcef5`; the ordered-lowerer auth `d63e7780`
at `fc819ff0` remains outside the protected/pinned dependency path. This gap is
separate from that dependency gate: the generic ordered registry does not by itself
define per-row authoring intent or invoke RLRMP's lowerers after matrix patches.

## Why the two axes are not leaf patches

Static source inspection shows that `force_filter_feedback` is first resolved in
`src/rlrmp/train/config_materialization.py:389-425`, then participates in the target
support contract at lines 523-526 and changes the controller input basis and
dimension. `src/rlrmp/train/run_spec_authoring.py:893-915` emits either a 6D
position/velocity/force basis or a 4D position/velocity basis. Lines 924-967 also
stamp the choice into training axes, distributions, and the broad-PGD payload.

Broad-epsilon PGD is built as a typed configuration at
`src/rlrmp/train/config_materialization.py:441-470`, constrained against the target
support at lines 528-548, and lowered into training mode, method payload, loss,
fidelity, task inputs, and worker execution contracts. Toggling only a copied `hps`
field after those derived fields exist would leave the compiled spec internally
inconsistent.

The only current tracked outer recipes with embedded Feedbax training specs are:

| Recipe class | Current examples | Relevant limitation |
|---|---|---|
| nominal C&S supervised | `results/cb3685a/runs/*.json` | force/filter visible only |
| broad-PGD supervised | `results/1ab1fef/runs/epsilon_scaled_short_3500to1000.json` | force/filter visible; soft-energy continuation semantics, not the matched nominal base |
| adaptive-epsilon continuation | `results/c6c5997/runs/*.json` | force/filter visible; different registered method and continuation contract |

There is no current content-pinned, force-hidden base that can be paired with those
without authoring a new compiled base. Reusing removed legacy stock is forbidden as
a runtime input. Splitting the family across unrelated historical bases would also
violate the requirement to hold task, plant, optimizer, budget, and seed fixed.

## Rejected escape

One could manually patch the graph input shape, graph descriptors, task inputs,
training distribution, method mode and payload, objective/loss/fidelity summaries,
worker phase program, artifacts, checkpoints, and metadata in every row. That would
duplicate compiler knowledge in experiment JSON, make the KPI look artificially
small or large depending on patch factoring, and bypass the registered lowerers.
This lane intentionally did not author such a matrix.

The prospective KPI input names `results/2cb6a58/runs/matrix.json`, but that path is
absent by design. A revision-pinned `marginal_cost.json` must not be generated until
a real governed matrix exists.

## Filed owning issue

[issue:5816bf0], **Re-lower governed scientific authoring intent for each training
matrix row**, is open and structurally blocks `2cb6a58`; it is also structurally
related to `509368b`. Its current scope is:

1. define a content-pinned compact authoring-intent base accepted by the governed
   matrix surface, without re-accepting legacy flat launch configs;
2. apply row axes to that intent before compilation;
3. invoke registered RLRMP science lowerers in deterministic order to produce the
   complete `TrainingRunSpec` for each row;
4. include the authored-intent hash, lowerer IDs/versions, resolved root, and row
   coordinates in planned run identity and emitted manifests; and
5. add a conformance test for the exact
   `force_filter_feedback x broad_epsilon_pgd_training` 2 x 2 family, proving 4D/6D
   graph and descriptor consistency without compiler-path patches in the experiment.

Resolution belongs to that owning authoring/compiler surface; no RLRMP
experiment-local shim is acceptable.

The canonical-tool-shaped draft at
`results/2cb6a58/notes/marginal_cost_input.json` is separately ignored by
`.gitignore` line 240. [issue:fddd87a] structurally blocks `2cb6a58` while it adds
normal role-based tracking for the required KPI input and report filenames. The
draft must not be force-added: doing so would hide the policy gap and create the
escape this audit is recording. It cannot yield a revision-pinned
`marginal_cost.json` until a real governed matrix exists and the whitelist issue is
resolved.
