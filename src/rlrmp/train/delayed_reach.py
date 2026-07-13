"""Science lowering owned by the delayed-reach capability."""

from __future__ import annotations

from typing import Any

from rlrmp.train.science_vocabulary import ScienceMode


def lower_science_mode(hps: Any) -> ScienceMode | None:
    """Lower delayed-reach capability into its run-spec mode."""

    return ScienceMode.DELAYED_REACH if bool(hps.delayed_reach.enabled) else None


__all__ = ["lower_science_mode"]
