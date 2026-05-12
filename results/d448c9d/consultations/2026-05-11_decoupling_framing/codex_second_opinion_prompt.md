# Second-opinion request: critique my synthesis of your prior consultation

You earlier produced a detailed analysis of the rlrmp project's regulator/tracker decoupling puzzle (full prior response at `/tmp/codex_response.txt` if you want to re-read it; key files: `/Users/mll/Main/10 Projects/10 PhD/rlrmp/results/410d7ac/notes/decoupling_acid_test_mvp.md`, `/Users/mll/Main/10 Projects/10 PhD/rlrmp/src/rlrmp/networks/linear_controllers.py`, issue `d448c9d`). Another agent (Gemini 3.1 Pro) gave a parallel analysis. I've synthesized both and arrived at a recommended path. **I want your honest critique** — do you agree this path will get us to the answer efficiently, or am I making a mistake somewhere? Push back hard where appropriate.

## My synthesis

**Where you and Gemini agree:**
- The MVP's `LinearTrackerController` is structurally degenerate — trivial x_nom = target makes it an affine LTV regulator-to-target, not a real trajectory tracker.
- A minimum viable tracker must parameterize `x_nom(t)` as a time-varying trajectory plus direction-dependent `u_ff(t, target)`.
- Warm-starting `u_ff` from inverse dynamics and `x_nom` from minimum-jerk (or LQR) is critical to avoid the K-only basin.
- GRU representational analysis — linearizing around its nominal trajectory to extract local `K_t = -∂u/∂x_feedback` and `u_0(t)` — is the cleanest test.
- The bimodality of flavor-B Δv replicates (~25% at +40 to +57%, the rest negative) should not be averaged over. Cluster instead.

**Where you push back harder than Gemini:**
- The GRU may not be a "tracker" — it may simply be solving a different optimization problem. Objective mismatch and adversary mismatch are leading alternative hypotheses, not afterthoughts.
- Δv alone is too narrow a signature. The better primary diagnostic is a Pareto plot: x-axis = nominal peak velocity / movement time, y-axis = induced gain / worst-case adversarial loss / local closed-loop sensitivity.
- Direction-dependent feedforward is essential for center-out reaches; a single world-frame `u_ff(t)` is structurally insufficient.
- The matched-objective ladder (Riccati → trained LTV regulator → true LTV tracker → affine recurrent → GRU) under identical plant/cost/horizon/adversary is the missing experiment. The empirical anchor: trained linear regulator gave Δv = +78% in the MVP while analytical Riccati gives +1-27% on the same regime — a 30× discrepancy suggesting the training objective is not the Riccati game.

## My recommended path

**Tier 1 — No new training (cheapest, highest leverage):**

1.1. **GRU affine decomposition.** Use `src/rlrmp/analysis/induced_gain.py` (verify what it actually computes — if it doesn't yet provide local-Jacobian extraction along a nominal trajectory, add that). Linearize baseline GRU and adversarial GRU at every timestep around their respective no-disturbance nominal trajectories. Extract `K_local(t)`, `u_0(t)`, residual nonlinearity, and the nominal/feedback power split `||u_0||² vs ||K_local · e||²`. Compare.

1.2. **Bimodality clustering.** Cluster the existing flavor-B replicates (from `results/c723082/` per your prior reference) by features: nominal velocity profile, final loss, adversarial loss, local feedback Jacobian norm, hidden-state trajectory PCA, readout norm. Test whether the positive-Δv minority is structurally distinct.

**Tier 2 — Linear architecture engineering under matched objective:**

2.1. **Fixed-direction sanity test.** 1 start, 1 target, no direction generalization. Train 4 variants under matched plant/cost/horizon/adversary: (a) LTV regulator, (b) true LTV trajectory tracker with `u_ff(t) = inverse_dynamics(x_nom(t))`, (c) K-frozen tracker (only u_ff trainable), (d) u_ff-frozen tracker (only K trainable). Warm-start all from teacher trajectories.

2.2. **Matched-objective Riccati round-trip.** Make the trained LTV regulator reproduce the analytical Riccati Δv (+1-27% target). If it gives ~+78% under that clean setup, our training objective is not the Riccati game and we need to fix it before architectural comparisons mean anything.

**Tier 3 — GRU under the same matched objective:**

3.1. Train a GRU with the *exact same* matched plant/cost/horizon/adversary/delay treatment as the Tier 2 setup — no extra losses, no catch trials, no noise. Run the same Pareto diagnostic. If GRU still gives Δv ≈ 0 while linear gives Δv > 0, decoupling-via-architecture is the explanation. If GRU gives Δv > 0 under matched objective, the prior result was objective mismatch all along.

**Reasoning for ordering:**
- Tier 1 requires zero new training and immediately tests whether the GRU empirically *does* decouple (without committing to any new architecture-engineering work).
- Tier 2 is the cleanest test of objective vs architecture: if the trained linear regulator can't reproduce analytical Δv on matched objective, the whole pipeline needs auditing before more downstream work.
- Tier 3 is the conclusive architecture test, but only meaningful AFTER Tier 1 + 2.

I would **not** start by building a non-trivial-x_nom tracker (both you and Gemini suggested it, but I think it's premature) — we don't yet know whether the puzzle is about architecture (tracker vs regulator) or objective mismatch. Tier 1 + 2 first, then decide whether Tier 3 needs the heavy architecture engineering.

## Questions for you

1. Does this ordering make sense, or do you see a faster path to the answer?
2. Is there a Tier 0 — a cheap analytical or computational test — I'm missing that would clarify everything faster?
3. What's the highest-priority *single* experiment if I could only do one thing?
4. Are there any flaws in this plan you spot — confounds, hidden dependencies, assumptions that don't hold?
5. Is `induced_gain.py` actually the right machinery for the GRU affine decomposition, or will I need to build something new? (You can `cat /Users/mll/Main/10\ Projects/10\ PhD/rlrmp/src/rlrmp/analysis/induced_gain.py` to check.)
6. Anything else I'm not asking that I should be?

Format: be concise. Bullet points and short paragraphs preferred over essay form. Push back specifically where you think I'm wrong.
