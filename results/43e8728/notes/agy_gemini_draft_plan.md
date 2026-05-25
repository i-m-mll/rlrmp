# Updated Central Plan: rlrmp cs2019-to-RNN Robustness Programme

## Executive Summary

This plan outlines the next major experiment for bridging the human motor control findings from cs2019 to trained recurrent neural controllers (RNNs/GRUs). Based on the synthesis in `0b1f109` and the GPT-5.5 Pro review, the core objective is to treat the initial phase as a strict **game-equivalence programme**, not merely a behavioral reproduction experiment. 

The goal is to first prove that our training pipeline can reproduce the analytical broad-epsilon H-infinity game (cs2019's theoretical model). Only once the formal mathematical game is verified using a linear regulator should we attempt to interpret a GRU's behavior and architectural constraints (such as decoupling nominal speed and feedback gain).

## Phase 1: Analytical "Game Card" Materialization

Before any new training is launched, the exact analytical target must be established. This serves as the contract for the same-game certificate.

**Required Deliverable:** An analytical "Game Card" artifact must be created, detailing:
- Exact state vector definition (including delay augmentation handling)
- Discrete-time convention and discretization
- Cost schedule, running penalties, and terminal cost
- Control penalty scaling
- Disturbance channel `B_w` definition
- Epsilon norm and time integration convention
- Gamma and critical gamma (`gamma_star`) convention
- LQR and H-infinity gains
- Nominal trajectories
- Analytical worst-case disturbance characterization (or equivalent closed-loop induced-gain characterization)
- Induced gain values
- Nominal Δv signature

**Implementation Check:** *STOP.* Once the Game Card is drafted, the implementing agent must stop, remind the user, and ask for explicit consent and review before proceeding to the training pipeline.

## Phase 2: Game-Equivalence Linear Round-Trip

This phase is the load-bearing gate. It validates that the gradient-based training pipeline correctly implements the H-infinity game from the Game Card.

**Adversary Form Uncertainty (Load-Bearing Formal Check):**
A critical unresolved formal choice exists between using an **open-loop trajectory epsilon** (practical PGD surrogate) versus a **state-dependent H-infinity adversary** (Riccati-implied disturbance).
*Decision Point:* These two checks should be split. 
- *STOP.* After the open-loop linear training is evaluated, pause to remind the user and ask for a decision/consent on whether to proceed with evaluating a closed-loop state-dependent adversary based on smart-model feedback.

## Phase 3: GRU Evaluation Under Certified Game

Only after the linear gate passes is a GRU trained under the identically certified broad-epsilon game. We observe whether the GRU behaviorally couples or dissociates speed and feedback gain.

## Phase 4: Restricted-Field Generalization Contrast

Following the broad-epsilon training, a matched contrast using restricted physical-field distributional exposure (e.g., curl fields) must be run. This determines what adversary-class broadening actually buys, holding plant, task, cost, architecture, and evaluation suite completely fixed.

## Subordinate Components

The following existing issues represent specific investigations and mechanisms that support this central plan. They should be wrapped as subordinate children of this umbrella plan:
- **`020a65b`**: Full-state ε adversary class matching C&S Eq. 13 H∞ disturbance.
- **`6ec6b19`**: Cost-schedule sweep (t/N)^α on Riccati and trained architectures.
- **`daa48c8`**: Bimodal-replicate analysis: cluster-conditioned Δv reporting across all training runs.
- **`1ad3c16`**: Principled choice of evaluation perturbation amplitude for robustness testing.
- **`63cec06`**: Deterministic / declarative standard-analysis pipeline for matrix experiments.
- **`8fcb6c7`**: Decoupling decomposition analysis: variance partition u_ff + K_local·error + residual.
- **`cf56e1e`**: Capacity-restricted RNN ablation: rank-1 to full GRU under matched ε adversary.
- **`f7b1b17`**: Constrained-RNN regulator: force u_t = K_RNN(h_t)·x_t, no u_ff channel.
- **`b41c940`**: Migrate RLRMP Feedbax models and recent artifacts to graph specs.

## Supplementary Analyses

- **`ac06736` Two-link arm with 6-muscle activation dynamics training pipeline:** Evaluates decoupling with realistic plant biomechanics to see if real physical constraints suppress decoupling. Fits as a follow-up if the GRU cleanly dissociates speed and gain.
- **`a5e1450` Cross-partial nabla^2_{theta delta} L diagnostic:** Investigates local loss geometry and gradient sensitivity for worst-case optimization. Fits as a diagnostic if PGD struggles to find strong worst-case adversaries.
- **`31043a5` Adaptation vs. robustness distinguishability in trained networks:** Explores whether trained responses represent within-trial adaptation or fixed robust policy. Fits as a follow-up trial-history / meta-policy experiment.
- **`a3edc0c` Supplementary experiment: formal comparison of BCS/DAI/PAI-ASF training methods:** Compares alternative adversarial generation methods for robustness training. Fits as a supplementary method comparison once the core adversary behavior is established.
- **`6d62018` Investigate appropriate motor noise scaling for speed-accuracy tradeoff:** Tunes noise parameters to balance optimal speed with task accuracy. Fits as a calibration step prior to extensive production plant sweeps.
- **`65156e8` Variable reach length: implications for adversarial training and SISU detection:** Analyzes how changing reach distance affects robust behavior and uncertainty modulation. Fits as a generalization bridge after the single canonical reach is solved.
- **`297260c` Umbrella: Part 1 analysis modules:** Coordinates analysis tools for early-phase modeling and baseline behaviors. Fits as standard evaluation tooling.
- **`0af472c` Umbrella: Part 2 analysis modules:** Coordinates analysis tools for later-phase adversarial behaviors and SISU effects. Fits as advanced evaluation tooling.

## Ops / Hygiene / Dependencies

- **`76d3a8e` RunPod operational lessons:** Captures stability and lifecycle best practices for managing long-running training instances.
- **`216b368` Pod setup fragility:** Addresses environment persistence issues to prevent Python and dependency loss when spot instances get preempted.
- **`a8ed10f` Modal serverless GPU integration:** Explores serverless compute options to bypass spot preemption entirely and simplify scaling job deployments.
- **`3bd407b` RunPod runbook: sed pattern in §4d eats prefix:** Fixes automated environment setup scripts that incorrectly alter and corrupt critical pyproject/uv dependency files.
- **`f7d40f1` RunPod direct-image pulls hit rate limit:** Provides solutions for container deployment blocked by Docker Hub anonymous rate limits.
- **`f350f58` RunPod runbook: specify ~/.local/bin/ install location:** Ensures the runpodctl CLI tool is correctly mapped to the user's path during automated setup.
- **`fdad09d` train_minimax.py write run.json:** Ensures run reproducibility by forcing exact configuration parameters to save alongside model checkpoints.
- **`e75ddd7` Strip PNG dumps from git history:** Keeps the git repository size manageable by cleaning up historical generated exploratory plots.
- **`2ef67ca` Part 2.5 legacy pre-migration training runs:** Archives legacy configuration records to preserve historical context from early training sweeps.
- **`2092cb5` Relocate legacy 2ef67ca-era eval_* scripts:** Reorganizes older evaluation scripts to conform to the project's modernized directory structure.
- **`6d5c906` Author misattribution correction:** Fixes an author citation error (Crevecoeur, Scott, & Cluff 2019) in all repository documentation.

## Issue-Disposition Notes

**Historical / Superseded (Do not use as central plan drivers):**
These issues represent prior iterations and are retained for historical record only.
- `35f64be`, `6f783fa`, `753508c`, `84ee4ff`, `ce34c2c`, `83fc5b5`.

**To Remain Open as Subordinate Children:**
- `020a65b`, `6ec6b19`, `daa48c8`, `1ad3c16`, `63cec06`, `8fcb6c7`, `cf56e1e`, `f7b1b17`, `b41c940`.

**To Comment On After Final Plan Adoption:**
Relevant coordination/phase umbrellas that need updating after this plan is finalized:
- `c99ad9d`, `4d38c15`, `b33e8da`.
