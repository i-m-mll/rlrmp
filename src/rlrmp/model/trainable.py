"""Trainable-parameter selectors for RLRMP staged-network controllers."""

from __future__ import annotations

from typing import Any


def staged_network_trainable_parts(net: Any) -> tuple[Any, ...]:
    """Return trainable subtrees for an RLRMP staged network component."""
    params = [net.hidden, net.readout]
    if hasattr(net, "h0_encoder"):
        params.append(net.h0_encoder)
    sisu_alpha = getattr(net, "sisu_alpha", None)
    if sisu_alpha is not None:
        params.append(sisu_alpha)
    return tuple(params)


def staged_network_trainable_paths(
    *,
    sisu_gating: str = "additive",
    initial_hidden_encoder: bool = False,
) -> list[str]:
    """Return semantic GraphSpec paths matching ``staged_network_trainable_parts``."""
    trainable = ["nodes.net.hidden", "nodes.net.readout"]
    if initial_hidden_encoder:
        trainable.append("nodes.net.h0_encoder")
    if str(sisu_gating) == "multiplicative":
        trainable.append("nodes.net.sisu_alpha")
    return trainable
