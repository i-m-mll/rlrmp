# Feedbax Implementation Parity

Issue: `020a65b`. Umbrella: `43e8728`. Feedbax issue: `be30d67`.
GraphSpec migration issue: `b41c940`.

Phase 2 uses a Feedbax-native discrete linear mechanics component rather than
an RLRMP-only shim. The mechanics node is intentionally small: it evaluates

```text
z[t+1] = A z[t] + B u[t] + B_w epsilon[t]
```

and projects the configured position/velocity slices to the usual Feedbax
effector output. For the C&S game-card instance, `A`, `B`, and `B_w` come
directly from `rlrmp.analysis.cs_game_card.build_canonical_game()`.

## State Map

The 48D C&S state is six 8D blocks:

| block | delay | indices | labels |
|---:|---:|---|---|
| 0 | 0 steps | 0-7 | `px`, `py`, `vx`, `vy`, `fx`, `fy`, `eps_x_int`, `eps_y_int` |
| 1 | 1 step | 8-15 | same physical order |
| 2 | 2 steps | 16-23 | same physical order |
| 3 | 3 steps | 24-31 | same physical order |
| 4 | 4 steps | 32-39 | same physical order |
| 5 | 5 steps | 40-47 | same physical order |

The lag delay is encoded in the supplied `A` matrix. There is no separate
Feedbax `Channel` node in this Phase 2 parity graph.

## Disturbance Map

The C&S disturbance input is 8D. Its matrix is fixed by Phase 0:

```text
B_w[:8, :] = I_8
B_w[8:, :] = 0
```

So an epsilon basis vector updates exactly one current physical coordinate and
does not write directly to any delayed lag block. The tests recover every
`B_w` column by stepping the Feedbax mechanics from zero state with a basis
epsilon.

## GraphSpec Scope

This is a small component-level GraphSpec integration: Feedbax Studio can
serialize and instantiate a `LinearStateSpace` node with explicit matrices,
input ports `force` and `epsilon`, and output ports `effector` and `state`.
It is not the full RLRMP controller/training GraphSpec migration. Issue
`b41c940` remains sequenced after Phase 2 and before Phase 3/4 so that the
controller, adversary, losses, rollout observables, and evaluation wiring are
represented as one modern graph rather than mixed helper code.
