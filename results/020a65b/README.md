# Feedbax C&S Plant Parity

Phase 2 of umbrella `43e8728` materializes the Phase 0 C&S analytical game card
as a Feedbax `LinearStateSpace` mechanics component. The tracked code keeps
`rlrmp.analysis.cs_game_card.build_canonical_game()` as the source of truth and
adds regression tests that the Feedbax component and GraphSpec path reproduce
the exact 48D `A`, `B`, and `B_w` update used by the Riccati reference.
