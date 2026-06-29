# Direct-epsilon soft-lambda redo

This issue directory holds the frozen direct-epsilon objective-level redo for
RLRMP issue 7180984. The materializer consumes the corrected HVP/Lanczos p90
lambda scale from `results/06a4dc8/canonical_soft_lambda_hvp.json`, evaluates
the c92 no-PGD frozen substrates at beta values `0.95`, `1.05`, `1.2`, `1.4`,
and `1.8`, and reports old hard-cap ratios only as sidecar diagnostics.
