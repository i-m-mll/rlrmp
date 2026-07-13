# Frozen legacy payload oracles

These files preserve the minimal immutable JSON bytes needed by schema, shape,
and tracked-payload regression tests after their originating experiment
directories are retired. They are test inputs only: runtime code must not load
from this directory.

| Fixture | Origin issue | SHA-256 |
|---|---:|---|
| `active_perturbation_response_manifest.json` | `020a65b` | `178ec2b9c42b42ef04ef0bccc56718a07d7a7f377fc72c523d83830fa02ec3ac` |
| `gru_evaluation_diagnostics.json` | `0203d1f` | `1b1028084b0476e48f5722a90449c38d76c874cba6f6af800cff2042a77384cc` |
| `guided_distillation_run_spec.json` | `9727d79` | `a91a18282b00d61ba290469d467424415e03db1a5febbcb03c5da4e9ca5424dd` |
| `historical_perturbation_response_manifest.json` | `b8aa38e` | `fff3228054e443f676475adab4272c83022f449ca7a57ef9c8cc1289bc6a2fbb` |
| `objective_comparator_sidecar.json` | `5f70333` | `bf333adfc57d74884005438e9ea7fe5d020403a57bc73451671bff392eeac704` |
| `sisu_perturbation_class_comparison.json` | `e4800d6` | `8cbe56994e0988ea7ea662f1545e865fa1cf5bbd536af50a1c1d4a48e2975d75` |
