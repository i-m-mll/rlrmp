# de51ee1 Resolution Addendum

Issue `de51ee1` traced the `gaussian_bump` minimax legacy-vs-native divergence
reported in `retro_equivalence_verdict.md` to native initial-slot construction:
the production native executor ignored `warmup_model`, while the legacy
`run_training` path loaded the explicit warm-start controller before entering
the adversarial loop.

The native path now loads the explicit warmup controller before constructing
controller slots and optimizer state. The regression
`test_minimax_native_initial_slots_honor_explicit_warmup_model` saves a
fixed-seed sentinel warmup model and verifies that native initial slots use that
controller instead of a fresh seeded controller.
