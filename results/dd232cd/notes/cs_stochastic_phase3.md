# Phase 3 Released Stochastic Rollout Recovery

Issue: `dd232cd`. Phase 3 issue: `7a459bb`.
Umbrella: `43e8728`.

Rerun metadata:

- Discretization: `euler`.
- Lane: `released_stochastic`.
- Lane scope: Released-code stochastic lane: Euler plant plus sampled sensory, motor/process, and signal-dependent control noise. Bellman parity is not claimed unless a separate stochastic objective is derived.

Scope: Small common-random-number Monte Carlo evaluation of deterministic Phase 3 rollout-recovery controllers under Euler plus sampled sensory, state-space motor/process, and signal-dependent state noise.

Non-goals: No initial-state jitter sweep, no process-noise scale sweep, and no stochastic Bellman parity claim.

Monte Carlo settings:

- Trials: `24`
- Seed: `2323`
- Motor covariance scale: `1e-08`
- Process covariance scale: `None`
- Signal-dependent scale: `0.02`
- Certificate gamma factor: `1.4`

Claims guardrail: Deterministic init labels identify source controllers only; this lane evaluates released-code stochastic forward simulation and does not derive or claim a stochastic Bellman objective. The deterministic certificate columns are re-audits of the same controller gains, not stochastic induced-gain certificates.

Source artifacts:

- `_artifacts/7a459bb/output_feedback_rollout_recovery/output_feedback_rollout_recovery.npz`
- `results/7a459bb/notes/output_feedback_rollout_recovery_manifest.json`

## Controller Matrix

| controller | source | cost mean | cost std | cost ratio | peak v mean | terminal err mean | action mismatch | deterministic gain err | exact L2 ratio | lambda/gamma^2 | finite-gamma feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| analytical_lqr_reference | analytical_lqr_reference | 4402.8469 | 56.27347 | 1 +/- 0 | 0.73270823 | 0.0034960293 | 0 +/- 0 | 0 | 1 | 1.5551183 | False |
| strong_optimizer_whitened__scratch | deterministic_phase3_scratch_fit | 4415.3428 | 44.572067 | 1.0028821 +/- 0.00552 | 0.73300631 | 0.0032077655 | 0.049956079 +/- 0.00633 | 0.97947196 | 1.1558731 | 2.0967916 | False |
| strong_optimizer_whitened_block_time__scratch | deterministic_phase3_scratch_fit | 4420.8427 | 43.453581 | 1.0041345 +/- 0.00566 | 0.73445002 | 0.0032837159 | 0.078159484 +/- 0.00433 | 0.97960462 | 1.158585 | 2.063401 | False |
| strong_optimizer_whitened__bellman_init | deterministic_phase3_preservation_init_fit | 4402.8468 | 56.274182 | 0.99999999 +/- 1.28e-06 | 0.73270819 | 0.0034960245 | 8.4849626e-05 +/- 1.82e-05 | 0.00013085276 | 0.99999243 | 1.5551095 | False |
| deterministic_bellman_initialization_raw | deterministic_phase3_initial_controller_only | 4403.1234 | 56.400518 | 1.0000625 +/- 0.000191 | 0.73254221 | 0.0035032015 | 0.0053847731 +/- 4.49e-05 | n/a | 1.00023 | 1.5552386 | False |

## Verdict

The released-stochastic evaluation keeps the deterministic Phase 3 interpretation: scratch-like fitted controllers remain behaviorally near in cost but still have substantial action mismatch relative to the analytical reference under the same sampled noise.
Best scratch stochastic cost ratio is 1.0028821 (strong_optimizer_whitened__scratch), with action mismatch 0.049956079.
The preservation-init fit remains indistinguishable from the reference at this Monte Carlo scale: cost ratio 0.99999999, action mismatch 8.4849626e-05.
The analytical reference mean cost is 4402.8469 with peak forward velocity 0.73270823.
This lane evaluates forward-simulation fidelity only; it does not add a stochastic Bellman objective or parity result.
