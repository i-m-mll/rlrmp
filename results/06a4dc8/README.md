Issue 06a4dc8 estimates the canonical corrected per-trial soft-lambda scale for
the c92 no-PGD frozen GRU substrates. The local materializer in
`scripts/materialize_canonical_soft_lambda_hvp.py` uses HVP-backed Lanczos
estimates of the largest algebraic Hessian eigenvalue of each per-trial
objective and reports `lambda_star_i = 0.5 * eigmax_i` under the ordinary
Hessian convention.
