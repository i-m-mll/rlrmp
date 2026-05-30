I have completed the read-only audit of the output-feedback lanes, focusing specifically on the covariance recursions and forward rollouts as you requested.

The audit completely validates your candidate hypothesis. 

### 1. Covariance Recursion Fidelity: Mathematical Match
I've performed a line-by-line comparison of `rlrmp`'s `robust_estimator_covariances` (`src/rlrmp/analysis/output_feedback.py`) against `minmaxfc_pointMass.m`. 

Despite the Python code looking different from the MATLAB, they are mathematically identical. 
- In C&S, `E` is initialized as a `1x8` vector `[1, 0, ..., 0]`. Therefore `E*E' = 1`, and the term `(E*E')^-1` drops out of the equation completely. `rlrmp` correctly recognizes this and just uses `H.T @ H`. 
- In C&S, `D = zeros(size(A))` but then sets the top `8x8` block to identity (`D(1:8, 1:8) = eye(8)`). So `D*D'` acts as a selector for the first 8 states. `rlrmp` correctly mirrors this with `Q_proc = plant.Bw @ plant.Bw.T`, because `Bw` injects process noise strictly into the physical state.

**Verdict**: The discrete update and covariance recursions match C&S precisely.

### 2. Forward Simulation Gap: Missing Stochastic Noise
As you suspected, `rlrmp`'s forward simulation is missing the C&S stochastic noise sampling.
- In `minmaxfc_pointMass.m`, C&S explicitly samples `sensoryNoise` and `motorNoise` using `mvnrnd` at every timestep, and computes signal-dependent noise `sdn` using `normrnd`. These are injected into the actual state propagation and observation matrices.
- In `rlrmp`, the functions `rollout_with_robust_estimator` and `rollout_with_kalman_estimator` only use the covariance parameters to compute the gains; they then propagate the trajectory completely deterministically, solely allowing for an explicit `epsilon` budget (which defaults to zeros during the clean LQR rollouts). They never sample empirical noise.

**Impact**: Because `rlrmp` evaluates cost deterministically, all of your artifacts and objective functions are tracking "clean" L2 trajectories. This means the metrics from `output_feedback_lane.md` and the robust Bellman tests are theoretically sound for a deterministic linear proxy, but they lack the empirical variance and noise inflation that the original C&S MATLAB model experienced. Crucially, the omission of `sdn` during the forward simulation means any neural network trained against these exact trajectories won't have the explicit incentive to suppress velocity that a multiplicative noise source normally induces (i.e. Flavor B).

I've documented the complete mathematical breakdown in an artifact here: [fidelity_audit_report.md](file:///Users/mll/.gemini/antigravity-cli/brain/3bcb0184-068d-4bf9-9e07-fba66c683756/fidelity_audit_report.md). 

Please let me know if you would like me to investigate the Feedbax/GraphSpec Phase 4 implications, check any other Bellman optimization files, or if you're ready to proceed with these findings!
