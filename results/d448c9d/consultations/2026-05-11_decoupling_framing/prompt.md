# Consultation: regulator/tracker decoupling and GRU robustness without velocity inflation

You are an expert in optimal/robust control theory and deep learning for motor control. I am orchestrating an ML research project (rlrmp) studying robust RNN controllers for point-mass reaching tasks. I am asking for your **independent opinion** on whether we are framing and approaching a specific puzzle correctly, and what the best path forward is.

## Project context

- **rlrmp** (the project): `/Users/mll/Main/10 Projects/10 PhD/rlrmp/` — JAX/Equinox/Optax stack. Trains RNN controllers for center-out reaching with H∞-style adversarial training.
- **feedbax** (dependency): `/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/worktrees/develop/` — the motor-control RNN training framework rlrmp builds on. (Note: feedbax's protected branch is `develop`, not `main`.)
- **Issue tracker**: git-bug (local, git-refs-based). Use `cd /Users/mll/Main/10\ Projects/10\ PhD/rlrmp && git-bug bug show <id>` to read an issue. The most relevant issues:
  - `d448c9d` — full-scope decoupling acid test (parent of MVP)
  - `410d7ac` — MVP child (what we actually did) — read its comment thread; the latest comments contain the corrected-MVP result and earlier retraction notes
  - `f695729` — phase umbrella ("regulator-coupling and biomechanical decoupling")
  - `03a148a` — replicate C&S 2019 analytical LQG vs H∞ Riccati
  - `4469cae` — eval-pipeline audit (Stage 1, prerequisite for some downstream work)
  - `c99ad9d` — training-methods coordination (cross-cutting decisions on training methods)
  - `4d38c15` — analyses coordination
- **Key code**:
  - `scripts/train_minimax.py` — main training entry (warmup + adversarial phases)
  - `src/rlrmp/networks/linear_controllers.py` — the linear regulator + tracker we built
  - `scripts/analyse_linear_decoupling_mvp.py` — Δv analysis script
  - `src/rlrmp/analysis/hinf_riccati.py` — standalone analytical H∞ / LQR Riccati solver (~2300 lines)
  - `results/410d7ac/notes/decoupling_acid_test_mvp.md` — corrected-MVP writeup (currently on branch `feature/linear-controller-mvp`, commits `45cc2db` + `dfac0d0`, auth pending)
- **Synthesis doc**: there's a long internal `synthesis.md` referenced from several issues; §4.2 (C&S Eq. 15 schedule), §5.3 (bimodal replicate analysis), §5.4 (analytical numbers), §7.1.1.

You may freely read the codebase, issues, results files, and synthesis doc. The session is in YOLO/full-auto mode.

## The puzzle (motivation)

Crevecoeur & Scott 2019 (henceforth C&S) study optimal H∞ control of a point-mass reach. They show analytically that, with a quadratic state cost `Q_t ∝ (t/N)^6` (back-loaded), the H∞-optimal controller (γ near γ*) exhibits **peak forward velocity inflation** — call this `Δv` — compared to the LQG limit (γ → ∞, no disturbance). The Δv is signed and projected on the reach axis. C&S Fig 1e shows Δv ≈ +10-15%. Our own analytical Riccati implementation reproduces this directionally: Δv ≈ +1-2.4% on rlrmp's regime, +1.0% on C&S's regime. **Sign agrees, magnitude is smaller**. Δv > 0 is treated as the canonical "robustness via velocity inflation" signature.

When we train *RNN controllers* (specifically a GRU with ~180 hidden, plus readout) under H∞-style adversarial training (parametric force-field adversary or full-state disturbance — depends on which run), we find:

1. **The GRU is robust**: it doesn't catastrophically fail under disturbance. Position error at the target stays small. The reach completes. (Specifically, "flavor-B" trained controllers — a specific minimax recipe used in the project.)
2. **But it does NOT show Δv > 0**: instead Δv ≈ −20 to −26% (group mean across replicates), with a bimodal distribution (~25% of replicates at +40 to +57%, the rest negative).
3. So the GRU achieves robustness *without* the analytically-predicted velocity inflation signature.

This is the puzzle. **Why does the GRU decouple robustness from velocity inflation, when an analytical optimal H∞ controller cannot?**

## The working hypothesis (regulator vs tracker decoupling)

Two reviewing agents (codex, gemini) earlier this year independently suggested the following framing:

- A **regulator** `u_t = -K_t · x_t` has only one control knob (K_t) which does both (a) drive nominal motion and (b) reject disturbances. Increasing K for robustness inevitably scales up control during normal motion → peak velocity inflated.
- A **tracker** `u_t = u_ff(t) - K_t · (x_t - x_nom(t))` has two channels: u_ff(t) carries the nominal motion plan and K_t only acts on deviations. So K_t can be made large for robustness without changing peak velocity.

**Hypothesis**: the GRU implicitly acts as a tracker — its hidden state encodes a nominal motion plan that feeds the output, while the recurrent dynamics provide deviation-based feedback. This would explain how it achieves robustness without inflating peak velocity.

## What we tried (MVP, issue `410d7ac`)

To test the hypothesis we built two **linear** architectures (~1.4K params each, training on laptop CPU):

- `LinearController`: `u_t = -K_t · e_t` with `e_t = (pos - target_pos, vel)` and K_t a learned LTV gain `(T × 2 × 4)`. This is a regulator.
- `LinearTrackerController`: `u_t = u_ff(t) - K_t · e_t` with K_t and u_ff(t) both learned LTV. This is the "tracker".

We trained **four matched models** (regulator/tracker × baseline/adversarial; same seed=42; warmup 1000 batches, adversarial 500 batches; `LinearDynamicsAdversary` ΔA·x at eta_max=0.1, 5 PGD inner steps). Δv was computed within each architecture as (peak_v_adversarial − peak_v_baseline) / peak_v_baseline.

**Result**: discriminator did **not split**.
- Δv_regulator = +0.775 ± 0.034 (SEM, n=5)
- Δv_tracker = +0.776 ± 0.034 (SEM, n=5)
- Tracker's |u_ff|_max: baseline 0.021 → adversarial 0.052 (4× increase but still ≪ K-driven feedback ~0.5)
- The adversary saturated its 0.1 Frobenius ceiling for both runs
- Both linear controllers show ~+78% inflation; the GRU baseline shows ≈0 inflation

## Our current realization (likely a flaw in the MVP)

The MVP's tracker has **trivial x_nom**: x_nom is implicitly the *constant* target state `(target_pos, 0)`, not a trajectory through state space. With constant x_nom:

```
u = u_ff(t) - K·(x - x_nom)
  = u_ff(t) - K·x + K·x_nom
  = [u_ff(t) + K·x_nom] - K·x
```

The tracker degenerates to "regulator-to-target + arbitrary time-varying bias `u_ff(t) + K·x_nom`". The free `u_ff` term carries no information about a planned path through state space — it's just an open-loop offset on top of the regulator. **No structural decoupling is possible** with this parameterization, regardless of training length.

To actually test the decoupling hypothesis, x_nom would need to be a *time-varying nominal trajectory* (e.g., a smooth minimum-jerk path from start to target), and u_ff(t) would need to encode the open-loop control that drives the system along that trajectory.

## Open questions for you (please address each)

1. **Is the framing right?** Is Δv (peak-velocity inflation between adversarially-trained and baseline-trained models of the same architecture) the right operational signature for "decoupling robustness from velocity inflation"? Is there a better signature we should be measuring instead — e.g., feedback-gain magnitudes, control-cost decomposition, sensitivity to disturbance frequency content, etc.?

2. **Is the linear-architecture MVP approach valid as a test of the decoupling hypothesis?** Given that the trivial x_nom collapse makes our MVP structurally degenerate, what is the *minimum viable* tracker parameterization that has a genuine chance to decouple? (Concretely: how should x_nom and u_ff be parameterized, what task structure is needed?)

3. **What is the best path forward to demonstrate decoupling (or rule it out)?** Lay out a concrete series of experiments — what to train, what to measure, what would constitute evidence for or against the hypothesis. Be specific about model architectures, training regimes, and analyses.

4. **Are there alternative hypotheses we should also consider?** Why else might a GRU show robustness *without* peak-velocity inflation? Some seeds for thinking (not exhaustive):
   - Sensorimotor feedback delay handled differently by RNN
   - GRU exploits the late-loaded `Q_t ∝ (t/N)^6` schedule by simply moving slower throughout (no need to inflate peak) — i.e. the "velocity inflation" is an artifact of how short the horizon is in the analytical setup
   - Nonlinearity allows a fundamentally different solution structure (e.g. bang-bang control)
   - The GRU is not actually robust in the same sense — maybe its "robustness" is on a different axis than what H∞ analytical theory measures
   - Adversary class mismatch (parametric force-field vs full-state ε): GRU might decouple against one but not the other
   - Stochasticity in training (catch trials, motor noise, sensory noise) shapes the optimizer's basin
5. **Anything we are missing?** Is there a literature you'd point us to, an analytical calculation we should do first, a representational analysis on the GRU we should run, or a setup change that would clarify everything?

## Format of your reply

- Be specific. Reference files, equations, and concrete experiments.
- Disagreement with our framing is welcome — push back hard if we are mis-framing.
- A bulleted plan (with priorities) is more useful than a long essay.
- Length: thorough but not exhaustive. ~1500-3000 words is fine. More if genuinely warranted.
