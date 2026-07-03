# Handoff: Feedbax-native alignment, next phase

Written 2026-07-03 by the coordinating agent that ran umbrella `64a04e0` to completion.
The reader is assumed to be a Fable-class agent assuming coordination of the successor
work. Orient from the Mandible ledger FIRST (`mandible umbrella status <id> --deep`);
this document supplies intent, principles, and operational knowledge the ledger doesn't.

## The principles (non-negotiable; the whole point)

1. **Data separate from code.** Generated/empirical/adopted data never lives as source
   constants. It lives in tracked specs or governed data products with schema versions,
   roles, and hashes; code holds schemas/loaders/builders. An AST lint enforces this.
2. **Contracts are obeyed, never bypassed.** If feedbax's API can't express a need, the
   fix is a feedbax change (issue → lane → auth), never an rlrmp-side shim, private-write,
   monkeypatch, wrapper, or "temporary" workaround. When a worker hits a gap it STOPS and
   reports; the coordinator routes the gap to the owning repo.
3. **Everything is an official component.** Trainers, analyses, diagnostics, adversaries,
   perturbations: either a generalized feedbax primitive (preferred whenever the concept
   is not project-specific) or an rlrmp component that participates fully in feedbax's
   registration/schema/migration contracts. No side systems, no parallel mechanisms.
4. **Components are parsimoniously modular graphs, not monoliths.** Refactor down to
   leaf components; generalized leaves live in feedbax; only genuinely project-specific
   science stays in rlrmp.
5. **Every run/eval/analysis is a schema-compliant, migratable spec.** Specs are the
   source of truth; execution materializes serialized specs (serialize-don't-re-derive);
   run records are natively-emitted manifests, never post-hoc reconstructions; extension
   payloads are governed families with versions and rejection rows.
6. **CI/tests prevent regressions structurally.** The `feedbax_contract` marked gate
   (151 tests at handoff; `ci/feedbax-contract-suite.toml` manifest with per-family
   minimum counts and negative canaries; skips are failures) encodes the residual classes
   we eliminated: retired-ID scan, descriptor canary, export-parity matrix, generated-data
   lint, re-accretion ratchet, write-surface custody (deny-by-default, generated branch
   matrix), lineage invariants, import boundaries, version pins, Lane B/C terminal gates.
   When you fix a residual class, extend the gate so it cannot return; a fix without a
   guard is half a fix.
7. **Residuals are fixed in-wave, not deferred.** Any bug/gap discovered during work is
   filed (own issue, adopted into the coordinating umbrella) and dispatched for fix in the
   same wave. Honest STOP-and-report from workers is prized; papering over is forbidden.
8. **The ledger is the coordination surface.** Every lane: issue, branch link, dispatch
   verification, coordinator diff review against acceptance, closeout checkpoint
   (worker_status/branch/blockers/next_action/delegation), narrative comment. Umbrella
   body children table kept current; manifest refreshed at dispatch points.

## State at handoff

- Umbrella `64a04e0` (52 children): final auth `eac2e99e` MERGED to rlrmp `main`
  (fdebf22), closing 30 children. Seven feedbax auths landed on `develop` across the
  campaign. rlrmp `main` was ~68 ahead of origin at handoff — user pushes.
- Full-suite state: rlrmp 1140 passed / 0 failed (tests/todo excluded: parked files
  needing `yaml`, pre-existing, 4 collection errors on old main reduced to 2);
  feedbax 1211+ passed / 0 failed. Marked gate 151/0.
- Predecessor umbrellas (all closed, merged): 09aed43 → 19553e9 → 588483d (analysis
  bundles/sidecars/regeneration/retention → feedbax) → 64a04e0 (graphs + training).
- Key docs: `results/64a04e0/notes/terminal_gate_readiness.md`,
  `results/f5d9695/notes/lineage_audit.md`, `results/e9fc384/notes/graph_sidecar_audit.md`,
  feedbax `docs/design/spec_successor_custody_audit.md`.

## The next-phase mandate (user-directed)

1. **Deliberately-open items:**
   - `00f97d5` + **deleting rlrmp parts 2–3 at the right time** (trigger issue `9134735`,
     feedbax-side parent `de19ed2`): retire the frozen parts-2/3 module workflows, then the
     two transitional feedbax analysis surfaces (`--batched` analysis flag; bridged legacy
     module manifests — inventory each bridged manifest: shim recipe or declared
     non-replayable). Also sweeps up `scripts/train_part2_5.py` (ratchet-inventoried,
     no owning child yet) and the two allowlisted docstring `feedbax._io` references.
     Timing is the user's call — confirm before deleting notebooks/parts.
   - `63cec06`: done but lifecycle-open on external `5c6b294` (path-resolvable seeded
     Studio run artifacts). Either drive `5c6b294` or keep the exception documented.
   - `e8452a4`: the induced-gain LTV/LTI/power-iteration math lift into feedbax
     (deliberately deferred; only the introspection fix landed). A real design lane —
     use a Fable subagent for the spec.
2. **The deletion stage** ("the stage that hasn't happened yet"): the compatibility
   adapters, legacy loaders, argparse shims, and allowlisted inventories are the explicit
   to-do list (`ci/legacy-pattern-allowlist.toml`, `ci/retired-component-id-confinement.toml`,
   `ci/write-surface-allowlist.toml` — shrink-only is ceremony-free and the guards prove
   completion). Also a dead-code sweep now that the gates define what's active. Expect a
   deletion-heavy diff; the user explicitly wants this. Sequence AFTER parts-2/3 removal
   where entangled; confirm with the user which compat paths they still exercise.
3. **Audit 588483d's completeness** (user lacks confidence in it): a THOROUGH residual
   sweep over the analysis system — diagnostics, post-run eval banks, materialization
   pipelines, sidecars, retention/regeneration — verifying every piece participates in
   the feedbax recipe/bundle/manifest contracts per the principles above. Authority
   granted: reopen `588483d`, reopen its children, add new children as needed. Method:
   sonnet/opus fan-out for archaeology (inventory every analysis surface, map each onto
   the contracts), Fable subagents for the gap analysis/spec judgments, Codex for fixes.
   The 7811e47-style pattern (scan + canary + gate registration) is the model for making
   the audit's conclusions permanent.

## Operational doctrine (hard-won; follow it)

- **Model tiering:** Codex (`gpt-5.5`, high reasoning) is the DEFAULT for all
  implementation lanes. Opus subagents ONLY on verified Codex usage exhaustion — and
  re-probe Codex after the stated reset time (the previous coordinator lapsed here once;
  the user corrected it). Sonnet for mechanical archaeology/inventory. Fable forks
  (`subagent_type: "fork"`) for mission-critical spec adjudication only.
- **Codex invocation:** `codex exec -m gpt-5.5 -c 'model_reasoning_effort="high"'
  --dangerously-bypass-approvals-and-sandbox -C <worktree> --output-last-message
  <log>.last.md - < <prompt-file> > <log> 2>&1`, one Bash background task per lane
  (NEVER batch several in one backgrounded shell — you lose per-lane completion events).
  `codex exec resume <session-id> ...` (no `-C`; cd first) continues a session for
  review-driven iteration; capture the session id from the log head.
- **Lane prompts must carry:** worktree path + branch + base, verified current facts
  (specs' file:line anchors go stale — say so), owned surfaces + sibling-ownership
  manifest, acceptance criteria + non-goals, env rules, Mandible process steps
  (status/claim/commit-linked/activate; workers never comment/merge/push/auth),
  stop-and-report conditions, report format. Coordinator reviews EVERY diff and
  independently re-runs key suites before closeout.
- **rlrmp env hazards:** worktree `.venv` symlinks the user's live shared venv — lanes
  must NEVER `uv sync/venv/pip`; tests via `PYTHONPATH="<worktree>/src" uv run --no-sync
  python -m pytest ... -p no:cacheprovider`. Bare `uv run` (without --no-sync) dirties
  `uv.lock` with editable-path churn — restore with `git checkout -- uv.lock` (agents did
  this repeatedly). `_artifacts/` is a shared symlink: read-only for lanes except
  explicitly-scoped additive writes. `.issue` conflicts on every merge (resolve ours).
  feedbax worktrees have LOCAL venvs (uv sync once is fine there).
- **Auth topology:** feedbax merges per-wave to `develop` (additive-only while rlrmp
  main depends on it); rlrmp accumulated on a master integration branch last campaign —
  next phase may merge to main per-wave instead if the user isn't mid-experiments; ASK.
  `mandible auth check` before submitting; auth check inspects the REPO ROOT worktree
  for dirt (a dirty root uv.lock blocks; align content into the branch, then the user
  clears the root copy). Stage `--closes-issue` per done child; verify the closure
  surface after submission. Bump `ci/feedbax-ref.toml` after each feedbax merge+push.
- **Known small caveats:** `product_identity_hash` gate family is pending_enrollment
  (owner 108b4d3; substance enforced elsewhere). `results/6cfa892/scripts/materialize_
  closed_loop_soft_lambda_redo.py` emits its own flat_rows-bearing JSON (within budget
  today; same double-serialization pattern as the remediated d469108 file — candidate
  for the same split). Many stale feature worktrees in both repos await `wt rm` cleanup
  (file/append a `Repo maintenance` issue). tests/todo needs `yaml` or retirement.
  This handoff file is untracked at write time — commit it via the first lane of the
  next phase (or the user commits it).

## Ledger anchors

Umbrella `64a04e0` (full history in comments; demarcation comment separates the
pre-reconstruction record). Related: `588483d`, `19553e9`, `09aed43` (closed umbrellas),
`de19ed2`/`9134735` (parts 2–3 removal), `e8452a4`, `5c6b294`, `63cec06`, `00f97d5`.
Coordination issues: `4d38c15` (analyses), `c99ad9d` (training methods), `b33e8da`
(phases — post a comment there when the next phase's umbrella is created), `1d9ae6f` (meta).
