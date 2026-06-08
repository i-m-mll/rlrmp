"""Compatibility wrapper for the provider-neutral packing benchmark."""

from __future__ import annotations

from rlrmp.packing_benchmark import *  # noqa: F403
from rlrmp.packing_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
