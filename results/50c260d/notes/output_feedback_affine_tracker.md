# Affine Tracker Output-Feedback Bridge

Issue: `50c260d`. Umbrella: `43e8728`.

Scope: Staged same-game affine tracker curriculum rows: clean scratch baseline, clean feedforward stage, reward-objective feedback and joint rows for Riccati-epsilon, state-eigenspectrum, observer-error, and mixed perturbation families, plus isolated supervised action-match diagnostics.

Non-goals: No GRU, robust/H-infinity training arm, direct teacher-cloning success claim, or old Delta-v decoupling acid-test criterion.

## Verdict

The staged feedback curriculum does not rescue nominal same-game recovery under the retained bounded materialization. The final mixed joint row has clean action mismatch `0.00198` and exact-L2 ratio `3.8`. The best reward-objective joint row is `affine_joint_state_eig` with clean action mismatch `9.53e-06` and exact-L2 ratio `0.97`. This verdict is based only on from-scratch/reward rows, not analytical action labels. The supervised diagnostic rows are present but excluded from this verdict (`2` rows).

## Same-Game Affine Bridge Rows

| row | objective | training distribution | diagnostic | objective ratio | feedforward rel err | gain rel err | clean action mismatch | under-eps action mismatch | exact L2 ratio | lambda/gamma^2 | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| affine_clean_scratch_baseline | reward_rollout | nominal | False | 24.71518 | 0.99999867 | 0.999182 | 0.97251523 | 0.97324407 | 11.574086 | 2.3532455 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_ff_clean_stage | reward_rollout | nominal | False | 1 | 0 | 0 | 0 | 4.0281717e-16 | 1 | 1.5551183 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| affine_fb_riccati_eps | reward_rollout | riccati_eps | False | 0.75004114 | 0 | 0.0055228079 | 0 | 0.40746078 | 7.242879 | 22.481856 | ABNORMAL:  |
| affine_joint_riccati_eps | reward_rollout | riccati_eps | False | 1.0271949 | 8.4239245e-05 | 0.0054585968 | 0.0010506904 | 0.21427241 | 3.0291135 | 8.6154088 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_fb_state_eig | reward_rollout | eigenspectrum_state | False | 0.98478213 | 0 | 0.015219818 | 0 | 0.049488931 | 0.94248166 | 1.3828868 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_joint_state_eig | reward_rollout | eigenspectrum_state | False | 1.5553435 | 3.4283317e-06 | 0.01491418 | 9.529068e-06 | 0.044521564 | 0.97017726 | 1.5917232 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_fb_observer_error | reward_rollout | observer_error_state | False | 0.99482334 | 0 | 0.018223799 | 0 | 0.060304556 | 0.91535677 | 1.5551251 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_joint_observer_error | reward_rollout | observer_error_state | False | 1.7341464 | 5.8292733e-06 | 0.018091413 | 5.6400219e-05 | 0.059625945 | 0.97663196 | 1.5552148 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_fb_mixed | reward_rollout | mixed | False | 0.80705421 | 0 | 0.0053593677 | 0 | 0.35169369 | 5.3631677 | 16.281435 | ABNORMAL:  |
| affine_joint_mixed | reward_rollout | mixed | False | 1.0073139 | 0.00012734652 | 0.0055712695 | 0.0019805651 | 0.24251649 | 3.7960064 | 11.123647 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| affine_feedback_action_match_riccati_eps | supervised_action_match | riccati_eps | True | 6.4810188e-19 | 0 | 4.2211876e-17 | 0 | 4.0327422e-16 | 1 | 1.5551183 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |
| affine_feedback_action_match_mixed | supervised_action_match | mixed | True | 6.5772865e-19 | 0 | 4.2211876e-17 | 0 | 4.0327422e-16 | 1 | 1.5551183 | CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL |

## Standard Certificate Rows

| row | lens | status | action mismatch | transition mismatch | value gap | Bellman residual |
|---|---|---|---:|---:|---:|---:|
| affine_clean_scratch_baseline | nominal_clean | full_standard_certificate | 1554.3 | 0.39298 | 79.4733 | 1.0504 |
| affine_clean_scratch_baseline | riccati_epsilon_response | full_standard_certificate | 85.6981 | 0.400492 | 79.6239 | 1.02905 |
| affine_ff_clean_stage | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_ff_clean_stage | riccati_epsilon_response | full_standard_certificate | 8.73771e-28 | 0 | 0 | 0 |
| affine_fb_riccati_eps | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_fb_riccati_eps | riccati_epsilon_response | full_standard_certificate | 266.002 | 0.109494 | 126.286 | 62.7531 |
| affine_joint_riccati_eps | nominal_clean | full_standard_certificate | 0.0545494 | 0.0866906 | 392.243 | 2.10885 |
| affine_joint_riccati_eps | riccati_epsilon_response | full_standard_certificate | 59.4068 | 0.0982141 | 75.4268 | 408.768 |
| affine_fb_state_eig | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_fb_state_eig | riccati_epsilon_response | full_standard_certificate | 19.1519 | 0.124797 | 18.6546 | 99.4108 |
| affine_joint_state_eig | nominal_clean | full_standard_certificate | 0.000103412 | 0.0614312 | 36.3332 | 6.093 |
| affine_joint_state_eig | riccati_epsilon_response | full_standard_certificate | 18.7853 | 0.110282 | 9.24966 | 7.09753 |
| affine_fb_observer_error | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_fb_observer_error | riccati_epsilon_response | full_standard_certificate | 83.2032 | 0.249337 | 3.78791 | 5.1004 |
| affine_joint_observer_error | nominal_clean | full_standard_certificate | 0.00020484 | 0.147002 | 143.941 | 20.127 |
| affine_joint_observer_error | riccati_epsilon_response | full_standard_certificate | 524.306 | 0.164523 | 10.6589 | 7.54928 |
| affine_fb_mixed | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_fb_mixed | riccati_epsilon_response | full_standard_certificate | 206.587 | 0.105611 | 92.2396 | 31.6964 |
| affine_joint_mixed | nominal_clean | full_standard_certificate | 0.127973 | 0.0880032 | 599.434 | 2.35293 |
| affine_joint_mixed | riccati_epsilon_response | full_standard_certificate | 24.993 | 0.0992725 | 120.429 | 62.2382 |
| affine_feedback_action_match_riccati_eps | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_feedback_action_match_riccati_eps | riccati_epsilon_response | full_standard_certificate | 8.93794e-28 | 1.10638e-32 | -1.33646e-15 | 9.28477e-30 |
| affine_feedback_action_match_mixed | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| affine_feedback_action_match_mixed | riccati_epsilon_response | full_standard_certificate | 8.93794e-28 | 1.10638e-32 | -1.33646e-15 | 9.28477e-30 |

## Failure Decomposition

| classification | rows |
|---|---:|
| mixed | 2 |
| not_failure | 10 |
| optimizer_basin | 12 |

The failure decomposition is the `c45adde` companion diagnostic. It explains
failed rows but does not replace the standard bridge gate.

<sup>1</sup> State-weighted action mismatch and Bellman-Hessian residual can match
exactly when the Bellman action Hessian is a scalar multiple of the action cost
geometry on that row. In that case they are the same evidence expressed through
two certificate views; they diverge when downstream value geometry weights
action directions differently.

<sup>2</sup> Gain mismatch is a diagnostic sidecar, not the bridge gate. The gate
is disturbance-relevant same-game behavior under the standard certificate
components.

## Historical Regulator/Tracker Comparison

This is not the old `410d7ac` / `d448c9d` tracker MVP. Those rows were a
Delta-v decoupling acid test with degenerate or trivial `x_nom` structure. The
50c260d row replays the analytical nominal output-feedback trajectory and
action sequence from the same C&S game, then optimizes only the decomposition
between feedforward command and estimated-state feedback correction. The success
criterion is preservation or scratch discovery under the standard bridge
certificate and failure decomposition, not a revived decoupling demonstration.

## Output Files

- Tracked manifest: `results/50c260d/notes/output_feedback_affine_tracker_manifest.json`
- Ignored bulk arrays: `_artifacts/50c260d/output_feedback_affine_tracker/output_feedback_affine_tracker.npz`
