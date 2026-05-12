# 410d7ac — Linear regulator vs tracker decoupling acid test (MVP)

MVP-scoped first-signal experiment under parent issue `d448c9d` and phase umbrella
`f695729`. Two laptop-CPU runs of tiny LTV controllers on the lit-replication
`post_nojerk` task — one pure regulator `u = -K_t · e_t`, one tracker
`u = u_ff(t) - K_t · e_t` — to test whether the tracker parameterisation
decouples open-loop feedforward drive from closed-loop feedback stiffness.

Architectures live in `src/rlrmp/networks/linear_controllers.py`; the training
script (`scripts/train_minimax.py`) gains `--hidden-type linear` and
`--hidden-type linear_tracker`, and `setup_task_model_pair` dispatches to a
`point_mass_linear_controller` body builder that swaps `SimpleStagedNetwork` for
the LTV controllers without touching feedbax. Time is tracked implicitly via a
per-step counter the controller maintains in its `NetworkState.hidden` channel
(no `task.add_input` plumbing was needed).

See `notes/decoupling_acid_test_mvp.md` for the headline finding.
