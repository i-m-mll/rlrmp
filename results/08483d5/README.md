This directory tracks planning and no-launch diagnostic artifacts for the
adaptive soft-adversary damage and phenotype-matching issue.

The general support documents are in `notes/`: candidate procedure specs, the
adaptive-lambda formalism complement, and the analytical damage-recursion
addendum. The output-feedback damage estimate and recursion comparison are also
tracked here as the first concrete damage-scale check; they are diagnostic
evidence, not a training launch or locked run spec.

Subsequent local no-launch diagnostics add a GRU PGD damage sanity check and a
frozen direct-epsilon adaptive-lambda replay. These are mechanism checks on
existing local artifacts only; they do not update controller weights or
authorize any launch.
