# Closed-Loop Finite Adversary Policy Design

This note defines the reusable finite policy contract for the closed-loop
soft-adversary lane.

## Policy classes

`linear_no_bias` is a shared time-varying linear policy:

```text
epsilon[t] = K[t] @ feature[t]
K.shape = (time, epsilon_dim, feature_dim)
```

The feature vector must be centered before evaluation. For the C&S full-state
basis, `target_centered_full_state_features` subtracts target position from the
position-like coordinates in each 8D mechanics-state block. Therefore zero
centered feature input implies exactly zero epsilon.

`affine` uses the same shared gains plus an explicit bias:

```text
epsilon[t] = K[t] @ feature[t] + b[t]
b.shape = (time, epsilon_dim)
```

The affine row is separate because `b[t]` can inject open-loop disturbance even
when feedback or error features are zero.

## Sharing and time indexing

One finite policy instance is shared across every trial in the frozen batch.
There is no per-trial independent policy parameterization. The policy may emit
different epsilons per trial only because live features differ by trial.

Time indexing is direct: parameters at index `t` are evaluated against live
features at rollout time `t`, producing epsilon at the same index. The primitive
does not define a hard projection as part of the scientific row; any cap belongs
to optimizer stabilization or audit reporting.

## Closed-loop semantics

The policy is evaluated during rollout from live perturbed state/error features.
It must not precompute a sequence from a clean rollout and replay that sequence
as open-loop epsilon. The first checked-in primitive intentionally exposes only
the policy classes, metadata, target-centered feature helper, and scalar audit
reporting hook. Wiring the primitive through the Feedbax rollout loop remains
the integration step before the rows can be treated as true closed-loop
training evidence.
