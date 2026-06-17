# Post-Run Eval Timing Continuation

Continuation context after the first optimization pass: local CPU only, one
`020a65b` H0 run, one rollout trial per replicate, twenty perturbation rows,
five feedback bins, worst-case epsilon with four PGD steps and three restarts,
and no raw bulk array writes.

The heavier subset keeps the same separate diagnostic bundles as the original
benchmark while adding enough perturbation rows and PGD work to exercise the
riskier candidates.

| Step | Adopted? | Total (s) | Standard | Eval diag | Figures | Objective | Map | Perturb | Feedback | Worst-case |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| opt5 serial backend control | yes | 69.01 | 11.93 | 1.94 | 3.25 | 24.57 | 4.09 | 6.84 | 13.14 | 3.25 |
| opt6 process-epsilon union | no | 68.37 | 11.52 | 1.92 | 3.17 | 25.23 | 4.26 | 6.50 | 12.90 | 2.86 |
| opt7 union plus staged PGD | no | 69.22 | 11.89 | 1.97 | 3.29 | 25.53 | 4.16 | 6.38 | 12.60 | 3.41 |
| opt8 adopted backends | yes | 65.65 | 11.11 | 1.87 | 3.02 | 23.78 | 3.98 | 6.60 | 12.34 | 2.97 |

The process-epsilon union attempt was rejected despite a small perturbation
runtime improvement. A serial-vs-union metric-tree check over the twenty-row
subset found large full-Q/R/Qf cost-summary differences for some process rows.
The likely cause is that replacing per-row graph-adapter identities with one
shared adapter changes stochastic graph key paths and no longer matches the
existing paired-base semantics.

The staged PGD backend was kept as an explicit opt-in because focused tests show
it matches the serial optimizer on deterministic objectives, but it is not the
default on this CPU path. On the heavier all-bundle subset it made the
worst-case bundle slower than serial (3.41 s staged versus 2.86 s serial in the
adjacent run).

A real-run smoke check also compared serial and staged worst-case audit outputs
on the benchmark run with the same four-step, three-restart settings; optimizer
summary fields and candidate cost summaries matched to `1e-10`.

Raw timing payloads:

- `postrun_eval_timing_opt5_serial_backend_control.json`
- `postrun_eval_timing_opt6_process_epsilon_union.json`
- `postrun_eval_timing_opt7_union_staged_pgd.json`
- `postrun_eval_timing_opt8_adopted_backends.json`
