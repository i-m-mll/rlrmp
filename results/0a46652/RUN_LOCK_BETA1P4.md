# Beta 1.4 soft-adversary run lock

Status: not launchable.

This lock was prepared for umbrella `0a46652` after correcting the soft-PGD
objective reduction and running local frozen-batch audits against the c92
open-loop no-PGD GRU rows. No controller weights were updated and no training,
pod acquisition, push, auth request, merge, or issue closure occurred.

## Lambda candidates

The local finite-direction curvature audit estimated one lambda scale per c92
open-loop no-PGD row, using beta `1.4` as `lambda_beta = beta^2 * lambda_star`.

| substrate row | lambda_star | beta=1.4 lambda |
|---|---:|---:|
| `open_loop_small` | 6.103397e6 | 1.196265812e7 |
| `open_loop_moderate` | 5.673930e6 | 1.112090280e7 |
| `open_loop_stress` | 6.1489695e6 | 1.205198022e7 |

The estimates are close enough that a shared scale is defensible for a future
run lock. The median candidates are `lambda_star = 6.103397e6` and
`lambda_beta = 1.196265812e7`.

## Frozen-batch audit verdict

| intended row | audit substrate | status | reason |
|---|---|---|---|
| corrected open-loop soft PGD | c92 `open_loop_small/moderate/stress` | no-lock | The selected direct epsilon improves the penalized objective, but every substrate hits the trust-region cap on 100% of sampled trials. That is cap-dominated, not soft-only. |
| closed-loop linear no-bias | c92 `open_loop_small/moderate/stress` | no-lock | The finite shared no-bias policy primitive is implemented, but the frozen optimizer accepted only the zero policy on all substrates. That does not test an active closed-loop adversary. |
| closed-loop affine | c92 `open_loop_small/moderate/stress` | no-lock | The finite shared affine policy primitive is implemented, but the frozen optimizer accepted only the zero policy on all substrates. That does not test an active closed-loop adversary. |

Detailed audit outputs are in:

- `results/0a46652/notes/soft_adversary_audit.json`
- `results/0a46652/notes/soft_adversary_audit.md`

## Minimal intended rows

These rows are the intended comparison set, but they must remain
`prepared_not_authorized` and non-launchable until the audit failures are
resolved.

| row id | mechanism | lambda | launch state |
|---|---|---:|---|
| `soft_pgd_beta1p4` | corrected open-loop direct-epsilon soft PGD | shared candidate `1.196265812e7` or row-specific lambda above | blocked: cap-bound audit |
| `soft_cl_linear_no_bias_beta1p4` | shared finite time-varying linear no-bias policy | shared candidate `1.196265812e7` or row-specific lambda above | blocked: zero accepted policy |
| `soft_cl_affine_beta1p4` | shared finite time-varying affine policy | shared candidate `1.196265812e7` or row-specific lambda above | blocked: zero accepted policy |

Before launch, either reduce/redefine the safety cap and rerun the direct
epsilon audit, or increase lambda enough that the corrected direct-epsilon
optimizer is not cap-dominated. For closed-loop rows, complete the Feedbax live
rollout integration and rerun the frozen policy audits until they accept a
nonzero policy with cap-bound fraction below 5%.
