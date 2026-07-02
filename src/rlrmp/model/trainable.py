"""Trainable-parameter selectors for RLRMP staged-network controllers."""

from __future__ import annotations

from typing import Any

from feedbax.runtime.graph import Graph


def staged_network_trainable_parts(net: Any) -> tuple[Any, ...]:
    """Return trainable subtrees for an RLRMP staged network component."""
    if isinstance(net, Graph):
        params = [
            net.nodes["cell"].cell,
            net.nodes["readout"].layer,
        ]
        h0_encoder = net.nodes.get("h0_encoder")
        if h0_encoder is not None:
            params.append(h0_encoder.layer)
        sisu_modulator = net.nodes.get("sisu_modulator")
        if sisu_modulator is not None:
            params.append(sisu_modulator.gain)
        return tuple(params)
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
