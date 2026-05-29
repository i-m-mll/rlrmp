You are an independent read-only auditor for the rlrmp C&S 2019 game-card fidelity correction.

Repo to audit:
- /Users/mll/Main/10 Projects/10 PhD/rlrmp/worktrees/integration__43e8728-bellman-rollout-results

Local C&S materials:
- Paper PDF: /Users/mll/Main/10 Projects/10 PhD/cs2019.pdf
- Released ModelDB code snapshot: /private/tmp/rlrmp_dd232cd_fidelity_audit/cs2019_modeldb
- ModelDB upstream: https://github.com/ModelDBRepository/258846

Relevant rlrmp issues:
- dd232cd: current fidelity audit/correction work unit.
- 43e8728: cs2019-to-RNN game-equivalence umbrella.
- cb98e58: original Phase 0 analytical game card.
- 83fc5b5: output-feedback estimator-in-loop lane.
- 97604a8: output-feedback gamma-feasibility sweep.
- 583d764: robust Bellman diagnostics/formal objective work.
- 7a459bb: output-feedback rollout recovery.
- 3fb0891: C&S fidelity discrepancies and interpretation notes.

Important current discovery:
The current rlrmp output-feedback rollouts use process/sensory covariance in estimator gain/covariance recursions, but the actual forward rollouts are deterministic:
- y_t = H x_t
- x_{t+1} = A x_t + B u_t + B_w epsilon_t
They do not sample the C&S released-code forward-simulation sensory noise, motor/process noise, or signal-dependent control noise. C&S minmaxfc_pointMass.m appears to sample sensoryNoise, motorNoise, and signal-dependent noise sdn during forward simulation, while also using noise covariances in extLQG/computeExtKalman/robust estimator recursions. Treat this as a known candidate gap, but do not stop there.

Task:
Do an exhaustive C&S fidelity audit. Compare the current rlrmp game-card/output-feedback/rollout/Bellman/Phase 1-3 code and tracked artifacts against the C&S 2019 paper and all released ModelDB MATLAB code. Look for any remaining infidelity or ambiguity that would affect Phase 0 game card, Phase 1 analytical/adversary checks, Phase 3 linear Bellman/rollout training, or future GRU interpretation.

Read-only constraints:
- Do not edit files.
- Do not create branches, commits, issue comments, auth requests, or run destructive commands.
- You may run read-only shell commands, tests, greps, and small local inspection scripts if useful.
- You may inspect Mandible issues with `mandible issue show <id>`.

Minimum surfaces to inspect in rlrmp:
- src/rlrmp/analysis/cs_game_card.py
- src/rlrmp/analysis/hinf_riccati.py
- src/rlrmp/analysis/output_feedback.py
- src/rlrmp/analysis/output_feedback_rollout_recovery.py
- src/rlrmp/analysis/robust_bellman.py
- tests/test_cs_game_card.py
- tests/test_output_feedback.py
- tests/test_output_feedback_rollout_recovery.py
- tests/test_robust_bellman.py
- results/cb98e58/notes/
- results/83fc5b5/notes/
- results/97604a8/notes/
- results/583d764/notes/
- results/7a459bb/notes/

Minimum C&S code to inspect:
- minmaxfc_pointMass.m
- AugRobustControl.m
- extLQG.m
- computeOFC.m
- computeExtKalman.m
- script_minmax_pointMass.m

Audit dimensions to cover:
1. Plant and discretization: A/B construction, tau/k/delta, mass/scaling, disturbance-integrator states, model-error DA handling.
2. Delay augmentation: state ordering, number of delay blocks, what is shifted, what H observes, how Q and H are augmented.
3. Cost schedule: Q_t construction, terminal/stage indexing, ramp exponent, cap, horizon nStep vs T, R/control cost convention.
4. Disturbance channel and gamma: D vs B_w, affected coordinates, epsilon norm/penalty, gamma search/selection, finite-gamma feasibility interpretation.
5. LQG/non-robust lane: extLQG iteration, Kalman gain computation, process/sensory/internal/signal-dependent noise terms, whether rlrmp simplifications are acceptable or require a separate lane.
6. H-infinity/robust output-feedback lane: robust Riccati recursion, estimator covariance recursion, persistent M(:,:,k) issue, time indexing, released-code-fidelity vs mathematical-clean variants.
7. Forward simulation: deterministic vs stochastic simulation, sensoryNoise/motorNoise/sdn sampling, perturbation timing, cost accumulation, output fields.
8. Phase 1 audits: whether exact fixed-controller/open-loop epsilon audits match the intended C&S object, and what they omit.
9. Phase 3 training: whether Bellman and rollout objectives are training the right object under the corrected card, and which previous results must be rerun.
10. Artifact/reporting fidelity: labels that could mislead future agents, e.g. calling a deterministic/covariance-only lane "C&S faithful" when it is missing stochastic simulation.

Return format:
- Executive summary: pass/fail and highest-risk fidelity gaps.
- Findings table with severity: blocker / important / minor / clarification.
- For each finding: C&S evidence with file/line/function, rlrmp evidence with file/function, why it matters, and recommended fix.
- Separate "analytical recursion fidelity" from "forward simulation fidelity".
- List which existing Phase 0/1/3 results must be rerun after correction.
- List any issues/comments you recommend the coordinator make, but do not make them yourself.
- List uncertainties that require another formal/theory review rather than implementation.

Be skeptical. The project has already had two fidelity misses in this area, so assume the current card may still be wrong until proven otherwise.
