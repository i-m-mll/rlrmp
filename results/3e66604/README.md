Reconstructed context: `3e66604` tracks the first C&S GRU optimizer-stabilization
screen under umbrella `bd57e8f`. The initial Modal A10 grid keeps the exact-objective
screen contract (`nn_hidden=0`, `cs_lss`, fixed 15 cm simple reach, C&S stochastic
rollout preset, 4D delayed position/velocity GRU observations) and crosses learning
rate `3e-3`/`1e-3` with global-norm clipping `1`/`5` at batch size `250`. Analytical
action and observation-to-action metrics are materialized as audit-only diagnostics;
they are not checkpoint or seed selection criteria.
