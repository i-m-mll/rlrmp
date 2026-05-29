Confirmed: C&S uses **forward Euler** (`A = eye + delta*A`, `B = delta*B`), while rlrmp uses **ZOH** (`expm`). Below is the audit.

---

# C&S 2019 Fidelity Audit (read-only, integration/43e8728)

## Executive summary

**Verdict: FAIL on labelling and forward-simulation fidelity. PASS on analytical Riccati form, delay augmentation, cost ramp, disturbance channel, persistent-index quirk, and estimator covariance recursion.**

The three highest-risk fidelity gaps:

1. **Forward-simulation fidelity (blocker for any "C&S-faithful Monte Carlo" claim).** The released C&S code samples `sensoryNoise ∼ N(0, Ω)`, `motorNoise ∼ N(0, Ξ)` with the integrator-row patch `Oxi(7:8,7:8)=Oxi(5:6,5:6)`, and **signal-dependent control noise** `sdn = Σ_j N(0,1) C_{sdn,j} u` with `C_sdn(:,i,i) = 0.1 B(:,i)`, plus a step force impulse at `i=14` (`currentX(7:8)=simdata.pert`). The rlrmp output-feedback lane consumes the noise covariances in the estimator gain/covariance recursions but **the actual `x_{t+1} = A x_t + B u_t + B_w ε_t` rollout is fully deterministic** (`ε_t = 0` by default, or a deterministic Riccati/PGD adversary). There is no sampling of `sensoryNoise`, `motorNoise`, or `sdn` anywhere in `output_feedback.py`, `cs_game_card.py`, or `output_feedback_rollout_recovery.py`. (Confirmed via grep.)

2. **Plant discretization differs (Euler vs ZOH).** `minmaxfc_pointMass.m:70-72` uses forward Euler `A = I + δA_c, B = δB_c` (and same for `Aest`). `cs_faithful_pointmass()` uses zero-order hold via `jsla.expm` on the augmented `[A_c | B_c | B_{w,c}]` block (`hinf_riccati.py:1041-1049`). For dt=10 ms and ‖A_c‖∼1/τ ≈ 15 (force-filter eigenvalue), the diagonal of `A_d[F,F]` is ≈0.849 (Euler) vs ≈0.859 (ZOH), and `B_d[F,u]` differs by ≈0.011 — both ~1% gaps that propagate into K, P, γ*, and Δv at the same order as the +7.5% peak-velocity inflation that the card highlights.

3. **The "C&S-faithful" / "canonical_for_cs2019_information_structure" labels overclaim.** The Phase 0 analytical card markdown and the `83fc5b5` output-feedback lane note both present themselves as a fidelity reference. Neither flags the Euler/ZOH gap, the missing stochastic forward simulation, the missing signal-dependent noise wiring, the missing extLQG iteration, or the missing `i=14` step perturbation. The `phase3.lqr_comparator_scope` field is the one place where the simplification is acknowledged ("simplified delayed Kalman baseline, not a full extLQG parity implementation with signal-dependent estimator noise terms"). That single caveat is not enough.

---

## Findings table

| # | Severity | Surface | Topic |
|---|---|---|---|
| F1 | **blocker** | output_feedback.py, cs_game_card.py | Forward sim deterministic; missing sensoryNoise/motorNoise/sdn sampling and `i=14` pert |
| F2 | **important** | hinf_riccati.py (`cs_faithful_pointmass`, `_zoh_discretize`) | ZOH discretization vs C&S forward Euler |
| F3 | **important** | output_feedback.py, robust_bellman.py | extLQG iteration not ported; LQG arm is a one-pass Kalman filter without SDN |
| F4 | **important** | results/cb98e58/notes, results/83fc5b5/notes | "C&S-faithful" / "canonical" labels overclaim; do not enumerate omissions |
| F5 | **minor** | hinf_riccati.py (`_build_cs_8state_pointmass`) | `Csdn` (signal-dependent process noise) absent from plant linearisation |
| F6 | **minor** | output_feedback.py (`process_covariance`) | `Oxi = 0.001 B B'` derived from ZOH-`B`, not from `δ·B_c` as in MATLAB; numerically close, not identical |
| F7 | **minor** | output_feedback.py (`measurement_covariance`) | `Omega = I * Oxi(5,5) * noise_scale` uses ZOH-derived Oxi(5,5); numerically close |
| F8 | **minor** | cs_game_card.py, output_feedback.py | Phase 1 "exact fixed-controller audit" maximizes L2-bounded `cost(ε)`, not the C&S H∞ closed-loop game — open-loop surrogate. Already noted in card prose; could be more prominent. |
| F9 | **minor** | output_feedback_rollout_recovery.py | Trains only the clean output-feedback **LQR** objective (`output_feedback_clean_objective`), not robust/H∞. Acknowledged in `non_goals`. |
| F10 | **clarification** | output_feedback.py (`OutputFeedbackConfig`) | `use_matlab_persistent_m_index=True` is correctly documented; rlrmp's flag is the right modeling choice. |
| F11 | **clarification** | hinf_riccati.py | Riccati recursion form matches MATLAB exactly (verified algebraically via Woodbury; LQR limit matches standard discrete form). |
| F12 | **clarification** | hinf_riccati.py (`cs_eq15_cost_schedule`, `apply_delay_distribution_to_schedule`) | Ramp values, delay-block Q distribution, R=I, and terminal Q_f all match C&S MATLAB. |
| F13 | **clarification** | hinf_riccati.py | `cs_faithful_pointmass(disturbance_integrator=False, delay_steps=0, tau=0.06)` is described as backward-compatible; the `0.06` here differs from MATLAB's `0.066` — fine as a *legacy* compatibility mode, but agents could be misled into thinking the pre-correction tau was C&S's. |
| F14 | **clarification** | results/cb98e58/notes | Δv=+7.49% at γ=1.05γ* is a deterministic closed-loop number, not an averaged Monte Carlo trace; it happens to be close to Fig 1e's ~+7.76%, which masks the lane gap. |

---

## Analytical-recursion fidelity (PASSES)

For every analytical object listed in cs_game_card.py / output_feedback.py / robust_bellman.py, I matched the formula against the released MATLAB code. The recursions are equivalent (after algebraic transformation), and the persistent-index quirk is correctly modelled with a flag.

- **Plant A,B,B_w**: `_build_cs_8state_pointmass` produces the same continuous-time matrices as MATLAB lines 63-67 (mass=1, k=0.1, τ=0.066). Integrator coupling `A_c[vel,ε_int]=I/mass` matches MATLAB `A(3,7)=A(4,8)=1` (which assumes mass=1). The `B_w_d = I_8` assignment matches MATLAB `D(1:8,1:8) = eye(8)` (D being the 48×48 selector after augmentation; rlrmp's (48,8) `B_w` with eye-on-top is functionally equivalent for `B_w ε`).
- **Delay augmentation** `_apply_delay_augmentation` matches `AugRobustControl.m`: top-left block = `A_phys`, identity blocks on subdiagonal, `H` selects oldest block, lag rows of `B`/`B_w` zero.
- **Q distribution** `apply_delay_distribution_to_schedule` matches `AugRobustControl.m:38-51` including the leading `h`-step pad with `Q[0]`.
- **Cost ramp** `cs_eq15_cost_schedule` matches MATLAB `runningalpha` line 28-30: `fact = min(1, ((t+1)/T)^6)` (Python 0-indexed = MATLAB 1-indexed shift), entries 0-3 ramped at 1e6/1e5, entries 4-7 constant 1, R=I_2.
- **H∞ Riccati backward recursion**: rlrmp's `P_t = Q + A.T @ P_next @ (I + M P_next)^-1 A` is algebraically identical to MATLAB's `M = Q + Aest' (M_next^-1 + BB' - γ^-2 DD')^-1 Aest` (verified via the identity `P(I+MP)^-1 = (P^-1+M)^-1`). The gain `K = (R+B'ΛB)^-1 B'ΛA` with `Λ = (I-γ^-2 P BwBw')^-1 P` reduces (R=I) to MATLAB's `L = B' M_next Λ_M^-1 A` with `Λ_M = I + (BB'-γ^-2 DD') M_next` (verified via `B'P(I+BB'P)^-1 = (I+B'PB)^-1 B'P`).
- **γ-admissibility predicate**: rlrmp's `eigvalsh(γ^-2 B_w^T P B_w) < 1 - 10^-6` is equivalent to MATLAB's `min(eig(γ^2 I - D' M D)) ≥ 0` (the latter is in n-dim; nonzero spectrum lives on the same m_w eigenspace because D selects the physical block).
- **Robust estimator covariance recursion** `robust_estimator_covariances` matches MATLAB lines 253-255: `Σ(t+1) = A (Σ(t)^-1 + H'(EE')^-1 H - γ^-2 Q(t))^-1 A' + DD'`. The `(EE')^-1` in MATLAB collapses to `1` because E=[1 0…]; rlrmp drops the term cleanly. `Q_proc = B_w B_w^T` matches MATLAB `DD'`.
- **Robust output-feedback gain** `robust_output_feedback_gains` matches MATLAB `u(i,:) = -B' (M(i+1)^-1 + BB' - γ^-2 DD')^-1 A (I - γ^-2 Σ(i) M(:,:,k))^-1 xhat` exactly. The persistent-index choice is correctly exposed via `OutputFeedbackConfig.use_matlab_persistent_m_index` and used as the default (matching released code).
- **xhat estimator update** in `_robust_estimator_rollout_arrays` matches MATLAB `currentXEst = Aest xhat + B u + A (Σ^-1 + H'H - γ^-2 Q)^-1 (γ^-2 Q xhat + H'(yx - H xhat))`. ✓
- **Initial state tile** `make_cs_output_feedback_initial_state` matches MATLAB `currentX = kron(ones(h,1), xinit)` (where MATLAB's `h = floor(δ/τ_delay)+1 = 6`, equivalent to rlrmp's `delay_steps+1 = 6` blocks of `xinit`).
- **B_w / D channel**: contractually identical (`B_w ε` writes into the physical 8-block only).

These items together justify the headline claim that **the analytical Riccati / robust-output-feedback / estimator-covariance lane is mathematically faithful to C&S** — except for the Euler vs ZOH discretization (F2).

---

## Forward-simulation fidelity (FAILS the "C&S faithful" label)

**F1 — Stochastic terms missing entirely.** The released `minmaxfc_pointMass.m` forward loop (lines 238-302) for the H∞ arm does:

```matlab
sensoryNoise = mvnrnd(zeros(8,1), Omega)';          % SAMPLED
motorNoise   = mvnrnd(zeros(48,1), Oxi)';            % SAMPLED, then patched
motorNoise(7:8) = zeros(2,1);
yx = H*currentX + sensoryNoise;                       % NOISY OBSERVATION
sdn = 0;
for isdn = 1:nc
    sdn = sdn + normrnd(0,1) * Csdn(:,:,isdn) * u(i,:)';   % SDN SAMPLED
end
currentX = Aest*currentX + B*u(i,:)' + D*wx + motorNoise + sdn;   % STOCHASTIC
```

The rlrmp counterpart `_robust_estimator_rollout_arrays` (output_feedback.py:707-718) does:

```python
y_t = H @ x_t                                          # CLEAN
x_next = plant.A @ x_t + plant.B @ u_t + plant.Bw @ eps_t   # DETERMINISTIC
```

with `eps_t` defaulting to zero (clean rollout) or coming from a deterministic Riccati feedback / PGD search.

Concrete consequences for downstream phases:
- The Phase 0B "output-feedback reference" (`analyze_phase0b_output_feedback`) reports deterministic peak-velocity / cost numbers. They are NOT comparable to C&S Fig. 1e or to any Monte Carlo run of `script_minmax_pointMass.m`.
- The Phase 1 audits (`exact_output_feedback_adversary_audit`) maximize cost over an L2-bounded `ε` in a deterministic estimator-in-loop. The resulting "worst-case cost" has no relation to the C&S simulation under sensoryNoise+motorNoise+sdn.
- The Phase 3 rollout-recovery experiments (`output_feedback_rollout_recovery.py`) train clean-rollout LQR objectives. They will not reveal whether the C&S H∞ controller is preserved under realistic noise; only whether the LQR equivalence holds in the clean limit.

This is the gap the task brief identified. Confirmed.

**F1.a — `i=14` step perturbation absent.** In `script_minmax_pointMass.m` the default `simdata.pert = [0;0]`, so this is a no-op for the default run. But:

```matlab
if i == 14
    currentX(7:8) = simdata.pert;
    currentZ(7:8) = simdata.pert;
end
```

is the standard machinery for the perturbation studies that produce the panels in Fig. 1d-e (cost vs perturbation). rlrmp has no equivalent in the output-feedback lane. If Phase 0/1/3 ever needs to compare cost-vs-perturbation curves to C&S, this hook must be added.

**F1.b — Signal-dependent noise absent from extLQG / LQR comparator.** `extLQG.m` is called with `Csdn` nonzero in the LQG arm of the script, so the LQG controller's gain matrix `L` and Kalman gain `K` are jointly tuned to account for control-magnitude-proportional process noise. rlrmp's `solve_lqr` is standard SDN-free LQR, and `kalman_estimator_gains` is a one-pass propagation of `Σ` for fixed K (it includes the SDN bookkeeping variable `s_temp` but ignores it: `_ = s_temp`). Combined with F1, the LQG arm in rlrmp is **not** the same controller object as C&S's LQG arm.

---

## Plant discretization (Euler vs ZOH)

**F2.** MATLAB `minmaxfc_pointMass.m:70-72`:
```matlab
A = eye(size(A)) + delta*A;       % forward Euler
Aest = eye(size(Aest)) + delta*Aest;
B = delta*B;
```

rlrmp `hinf_riccati.py:1041-1049` (`_zoh_discretize`) uses `expm`. The resulting A_d and B_d differ at O(δ² ‖A_c‖²) ≈ 2e-2. The dominant divergence is the force-filter row: `A_d[F,F] = 0.8485` (Euler) vs `0.8594` (ZOH); `B_d[F,u] = 0.1515` (Euler) vs `0.1406` (ZOH).

Two practical effects:
- Reported γ\* (≈9041 in rlrmp's card) is the boundary on the **ZOH** plant. The MATLAB script's γ-optimizer (with initial guess 50000) converges on the **Euler** plant. The two boundary values are slightly different; equality is not guaranteed.
- Δv inflation magnitude depends on which discrete plant is used. The +7.49% reported in the card is the *ZOH*-plant LQR-vs-H∞ Δv, not the Euler-plant equivalent.

This is not necessarily wrong (ZOH is the principled choice for small δ), but presenting the result as "C&S faithful" obscures the discretization choice.

---

## LQG / extLQG fidelity (FAILS in non-trivial ways)

**F3.** The rlrmp output-feedback "LQG comparator" is a one-pass discrete Kalman filter on top of standard discrete-time LQR gains, with `kalman_estimator_gains` (output_feedback.py:210-249) effectively dropping the extLQG inner-loop iteration. Specifically, the rlrmp code:
- Maintains `Sigma_e`, `Sigma_x`, `Sigma_ex` recursively but **does not iterate** the gain/filter pair (no outer `while` corresponding to extLQG's iteration to fixed point).
- Builds `s_temp = Sigma_e + Sigma_x + Sigma_ex + Sigma_ex.T` then discards it (`_ = s_temp`). MATLAB uses this to add `statedn = Σ_j D(:,:,j) sTemp D(:,:,j)'` to the innovation covariance (computeExtKalman.m:24-30). With `D=0` in the LQG call (`extLQG(Aest, B, Csdn, 0*H, H, ...)`), this would be zero; but the `Csdn`-driven SDN term inside `S = H Σ_e H' + Ω + statedn` would still pick up nonzero contributions through `computeOFC`'s `sdn` term.
- Does not implement `computeOFC`'s SDN-augmented gain: `L = (R + B' Sx B + Σ_j C_j' (Sx+Se) C_j)^-1 B' Sx A`.

In the deterministic limit (no SDN, no noise sampling) these simplifications are inert and the rlrmp lane is correct. Under stochastic forward sim with SDN, the resulting Kalman gains and LQR gains both differ from C&S.

The note (`output_feedback.py:2186-2189`) flags `"simplified delayed Kalman baseline, not a full extLQG parity implementation with signal-dependent estimator noise terms"`. Severity is **important** because the only acknowledgement is one inline string in a deep field; downstream agents reading the markdown render will not see it prominently.

---

## Notes/labelling fidelity

**F4.** Surveying tracked notes:

- `results/cb98e58/notes/analytical_game_card.md` is described as "the auditable C&S-faithful H-infinity target for the first cs2019-to-RNN game-equivalence gate" and "fixes the analytical game". It does NOT mention the ZOH-vs-Euler discretization, missing stochastic forward simulation, missing SDN, or missing extLQG iteration. A future agent reading this will assume parity with the released MATLAB code.

- `results/83fc5b5/notes/output_feedback_lane.md` (rendered by `output_feedback.py:render_markdown`) does call out the persistent-M index choice (`Robust command indexing: MATLAB-compatible: released C&S code applies M(:,:,k) after the backward loop`) and notes Phase 4 implications. But it markets the lane as "canonical for cs2019 information structure" without enumerating omissions.

- `results/97604a8/notes/output_feedback_gamma_sweep.md` and `results/583d764/notes/robust_bellman.md` are diagnostic in tone and do not overclaim fidelity beyond what they implement.

- `results/7a459bb/notes/output_feedback_rollout_recovery.md` correctly scopes itself to clean LQR rollout recovery and notes the non-goals.

`results/3fb0891/notes/` does not exist in this worktree (the brief lists `3fb0891` as the discrepancy-tracking issue, but no on-disk note exists; it lives only in Mandible).

---

## Auxiliary findings

- **F5 (minor).** rlrmp's `_build_cs_8state_pointmass` never constructs the C&S `Csdn` array; nothing in the analytic stack references SDN. This is consistent with F3 — the entire SDN modelling is absent. The audit doc `flavor_ab_review/findings/cs_alignment_audit.md` referenced from the docstring exists outside the repo (`/tmp/...`); future agents won't have access.
- **F8 (minor).** The "exact fixed-controller adversary audit" in `output_feedback.py:538-658` solves `max ε^T H ε + 2 g^T ε s.t. ‖ε‖≤r` (L2 trust region). This is **NOT** the C&S H∞ game. The card prose calls this out (and the `analyze_output_feedback_gamma_sweep` is the right way to upgrade to a γ-penalized check), but the label "exact" without "open-loop surrogate" might mislead.
- **F13 (clarification).** `cs_faithful_pointmass` docstring (hinf_riccati.py:541-544) suggests "`cs_faithful_pointmass(disturbance_integrator=False, delay_steps=0, tau=0.06)` reproduces the pre-9a0558e 6-state, no-delay form exactly." That earlier τ=0.06 is *legacy*, not C&S — the C&S value is 0.066. The phrasing could read as legitimizing 0.06.

---

## Existing results that must be rerun after correction

If the project decides to close the F1 gap by adding stochastic forward simulation (`sensoryNoise`, `motorNoise`, `sdn`, plus `i=14` pert hook):

- **All Phase 0B reference numbers** under `83fc5b5` notes (lqr/hinf peak forward velocity, costs, estimation error RMS). Currently deterministic.
- **All Phase 1 exact-fixed-controller audits** under `83fc5b5` notes and the `97604a8` gamma sweep. The L2 budget is currently derived from a deterministic worst-case Riccati ε; under stochastic forward sim, the right budget is either the realized Monte Carlo ε energy or a γ-penalty (which is already implemented as the diagnostic).
- **All Phase 3 rollout-recovery results** under `7a459bb` notes. Clean LQR equivalence under deterministic estimator-in-loop is trivial (innovation = 0). Under stochastic noise, the equivalence no longer holds and the experiment becomes meaningful.

If the project decides to close the F2 gap (Euler vs ZOH):

- **Phase 0 game card** under `cb98e58`: γ\*, K, P, Δv at every gamma factor. Order-1% changes expected.
- **Phase 0B/1/3** under `83fc5b5`, `97604a8`, `583d764`, `7a459bb`. All downstream numbers shift.

If the project decides to keep the analytical reference deterministic but add an explicit "C&S Monte Carlo" lane alongside (the cleaner option):

- No existing analytical results need rerunning; they remain valid in the deterministic sense.
- All "C&S faithful" / "canonical for cs2019" labels must be qualified to "deterministic analytical reference" or "C&S analytical recursion only".
- The four notes above must add explicit "what this lane does NOT match" sections.

---

## Recommended coordinator actions (do not make them; for the user to consider)

- **`dd232cd`** (current fidelity audit work unit): record this audit's specific findings (Euler/ZOH, missing stochastic forward, missing SDN/extLQG iteration, missing `i=14` hook) so they are tracked and not relitigated.
- **`3fb0891`** (C&S fidelity discrepancies): cross-link to the four note paths called out in F4 and propose either (a) re-label or (b) implement the stochastic forward lane.
- **`83fc5b5`** (output-feedback): add a short note enumerating what the lane does NOT do, since the lane is otherwise marketed as canonical.
- **`cb98e58`** (Phase 0 card): add a discretization-choice line (Euler vs ZOH) and a "deterministic analytical reference" qualifier to the headline claim.
- **`c99ad9d`** (training-methods coordination): note that any Phase 3 training meant to reproduce C&S signatures under realistic noise will need the stochastic forward lane and SDN-aware controllers before the comparison is meaningful.
- **`4d38c15`** (analyses coordination): flag that Phase 1 L2-budget audits are open-loop surrogates for the closed-loop H∞ game (already noted in the card prose, but worth a tier line).

---

## Uncertainties that need formal/theory review (not implementation)

- Whether ZOH or Euler is the "right" discretization choice for matching C&S Fig. 1e quantitatively. C&S's text doesn't say; the released code uses Euler. The +7.5% match on the ZOH-discretized rlrmp plant may be a coincidence rather than confirmation.
- Whether the persistent-index choice `M(:,:,k=1)` in the released code is intentional (a quasi-stationary approximation) or a bug. Both rlrmp's formal and persistent variants exist; the right experimental target depends on the answer.
- Whether the "B_w supports only the physical 8-block" (i.e., `Bw[8:,:]=0`) is the correct H∞ adversary class given that C&S's `D` is the same selector and only acts when `DA ≠ 0` (model-error disturbance). The formal H∞ game on the augmented state with this restricted `B_w` is a sub-game of "ε on the full augmented state"; the corresponding γ\* is conservative.
- Whether `simulate_closed_loop` in Phase 0 should be considered a faithful analytical object or just a sanity check, since C&S's deterministic-controller-plus-noise simulation is the only Δv comparison they actually report numbers for.
- Whether the rlrmp `cs_faithful_pointmass` integrator-coupling row `A_c[vel, ε_int] = I/mass` is the correct continuous-time analogue of MATLAB's `A(3,7)=A(4,8)=1`. The MATLAB code is mass=1; a future agent who tries mass≠1 will hit a parametrization choice that has not been validated against C&S theory. (Currently fine because mass=1 throughout.)
