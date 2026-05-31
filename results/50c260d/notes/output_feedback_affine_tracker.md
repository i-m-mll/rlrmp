# Affine Tracker Output-Feedback Bridge

Issue: `50c260d`. Umbrella: `43e8728`.

Scope: Same-game affine tracker bridge rows: reference replay, feedforward-only with K_ref frozen, K-only with u_ref frozen, both-from-scratch, spline tracker K_t, and selected state-coverage variants.

Non-goals: No GRU, robust/H-infinity training arm, direct teacher-cloning success claim, or old Delta-v decoupling acid-test criterion.

## Verdict

Reference affine replay preserves the analytical output-feedback policy (clean action mismatch `0`). The scratch rows are retained as discovery diagnostics; the best retained scratch clean-action mismatch is `0` on `gain_only_u_ref_frozen`. Treat any scratch rescue claim as provisional unless the standard-certificate and failure-decomposition rows agree.

## Same-Game Affine Bridge Rows

| row | training distribution | objective ratio | feedforward rel err | gain rel err | clean action mismatch | under-eps action mismatch | exact L2 ratio | lambda/gamma^2 | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| reference_affine_replay | none | 1 | 0 | 0 | 0 | 4.0281717e-16 | 1 | 1.5551183 | reference replay; no optimizer |
| feedforward_only_k_ref_frozen | nominal | 1 | 2.0269381e-08 | 0 | 2.2092888e-08 | 2.2177023e-08 | 1 | 1.5551183 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| gain_only_u_ref_frozen | nominal | 8.9409629 | 0 | 0.99194428 | 0 | 0.013101179 | 1.1701899 | 2.2693147 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| both_from_scratch | nominal | 5.6332398 | 0.99999852 | 0.99726568 | 1.6649674 | 1.6693673 | 3.0666062 | 2.221642 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_tracker_r20 | nominal | 5.8220608 | 1.0000001 | 0.9977714 | 1.7028431 | 1.7069708 | 3.0092886 | 2.2490431 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| both_from_scratch__state_eigenspectrum_m4_s1_w0p1 | eigenspectrum_state | 5.4398021 | 0.99995523 | 0.99880509 | 1.6792876 | 1.6841203 | 10.648311 | 2.3776026 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| both_from_scratch__state_eigenspectrum_m4_s3_w0p1 | eigenspectrum_state | 2.2677762 | 0.99995655 | 0.99879384 | 1.4129925 | 1.4169523 | 11.156593 | 2.3412709 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| both_from_scratch__observer_error_state_m1_s0p3_w0p1 | observer_error_state | 5.816406 | 0.999968 | 0.99919863 | 2.2161693 | 2.2242166 | 7.5959814 | 2.4083486 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_tracker_r20__state_eigenspectrum_m4_s1_w0p1 | eigenspectrum_state | 4.7158391 | 0.99996984 | 0.99692571 | 1.8787345 | 1.8864851 | 9.4782431 | 2.3289405 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_tracker_r20__state_eigenspectrum_m4_s3_w0p1 | eigenspectrum_state | 2.1106908 | 0.99998385 | 0.99804212 | 1.6094999 | 1.6140237 | 9.1451335 | 2.2721472 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |
| spline_tracker_r20__observer_error_state_m1_s0p3_w0p1 | observer_error_state | 5.0441482 | 0.99997777 | 0.99785815 | 2.4373678 | 2.4461345 | 6.5668835 | 2.2859019 | STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT |

## Standard Certificate Rows

| row | lens | status | action mismatch | transition mismatch | value gap | Bellman residual |
|---|---|---|---:|---:|---:|---:|
| reference_affine_replay | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| reference_affine_replay | riccati_epsilon_response | full_standard_certificate | 8.73771e-28 | 0 | 0 | 0 |
| feedforward_only_k_ref_frozen | nominal_clean | full_standard_certificate | 4.00735e-08 | 0 | 0 | 0 |
| feedforward_only_k_ref_frozen | riccati_epsilon_response | full_standard_certificate | 2.23089e-10 | 0 | 0 | 0 |
| gain_only_u_ref_frozen | nominal_clean | full_standard_certificate | 0 | 0 | 0 | 0 |
| gain_only_u_ref_frozen | riccati_epsilon_response | full_standard_certificate | 1.38841 | 0.559775 | 3.27907 | 0.831201 |
| both_from_scratch | nominal_clean | full_standard_certificate | 3720.97 | 0.151132 | 40.4048 | 10.031 |
| both_from_scratch | riccati_epsilon_response | full_standard_certificate | 87.2408 | 0.148587 | 40.4516 | 8.28214 |
| spline_tracker_r20 | nominal_clean | full_standard_certificate | 1374.28 | 0.140481 | 127.692 | 2.8561 |
| spline_tracker_r20 | riccati_epsilon_response | full_standard_certificate | 121.701 | 0.139313 | 127.725 | 2.39396 |
| both_from_scratch__state_eigenspectrum_m4_s1_w0p1 | nominal_clean | full_standard_certificate | 4.77673e+06 | 0.315898 | 12.8764 | 1.46433 |
| both_from_scratch__state_eigenspectrum_m4_s1_w0p1 | riccati_epsilon_response | full_standard_certificate | 26343.6 | 0.327981 | 12.9001 | 1.4278 |
| both_from_scratch__state_eigenspectrum_m4_s3_w0p1 | nominal_clean | full_standard_certificate | 1.01742e+06 | 0.270535 | 13.0354 | 1.30737 |
| both_from_scratch__state_eigenspectrum_m4_s3_w0p1 | riccati_epsilon_response | full_standard_certificate | 3243.18 | 0.283867 | 13.0617 | 1.28438 |
| both_from_scratch__observer_error_state_m1_s0p3_w0p1 | nominal_clean | full_standard_certificate | 129873 | 0.179081 | 8.52604 | 1.20337 |
| both_from_scratch__observer_error_state_m1_s0p3_w0p1 | riccati_epsilon_response | full_standard_certificate | 2451.53 | 0.186842 | 8.52954 | 1.20988 |
| spline_tracker_r20__state_eigenspectrum_m4_s1_w0p1 | nominal_clean | full_standard_certificate | 2.65978e+07 | 0.222375 | 8.39153 | 277.915 |
| spline_tracker_r20__state_eigenspectrum_m4_s1_w0p1 | riccati_epsilon_response | full_standard_certificate | 407984 | 0.236702 | 8.40576 | 277.363 |
| spline_tracker_r20__state_eigenspectrum_m4_s3_w0p1 | nominal_clean | full_standard_certificate | 526207 | 0.233585 | 10.5393 | 1.27049 |
| spline_tracker_r20__state_eigenspectrum_m4_s3_w0p1 | riccati_epsilon_response | full_standard_certificate | 2583.78 | 0.245714 | 10.5535 | 1.25574 |
| spline_tracker_r20__observer_error_state_m1_s0p3_w0p1 | nominal_clean | full_standard_certificate | 3.23681e+06 | 0.146916 | 5.7257 | 72.9052 |
| spline_tracker_r20__observer_error_state_m1_s0p3_w0p1 | riccati_epsilon_response | full_standard_certificate | 50046.9 | 0.154111 | 5.7344 | 72.6776 |

## Failure Decomposition

| classification | rows |
|---|---:|
| not_failure | 4 |
| optimizer_basin | 18 |

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
