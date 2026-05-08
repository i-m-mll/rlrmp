# Flavor-A vs Flavor-B adversarial training

This directory holds the cross-cutting synthesis and review of the **flavor-A
(input-instance, additive-force) vs flavor-B (model-class, structural `ΔA`)
adversary distinction** that runs through the rlrmp project's robust-control
work. The motivating empirical anchor is the failure of flavor-A adversarial
training in Part 2.5 to reproduce the Crevecoeur, Cluff, & Scott (2019)
peak-velocity inflation signature, and the recent Riccati-side diagnosis
(`tests/test_hinf_riccati.py::test_cs_faithful_qr_velocity_inflation`,
xfailed) that production's `B_w` is most likely flavor-(a). The first
flavor-B training run (`LinearDynamicsAdversary`, issue `c723082`, merged via
`2e21833`) and the analytical Riccati flavor-(b) extension (issue `97c227a`,
open) close the loop on the comparison.

The narrative artifact is `synthesis.md` (this dir). Heavy outputs and run
checkpoints live under `_artifacts/flavor_a_vs_b/` per the role-based
artifacts policy. Cross-refs: training-methods coord `c99ad9d`, analyses
coord `4d38c15`, induced-gain analyser issue `74bfd86`, Part 2.5 phase
artifact `results/part2_5/README.md`.
