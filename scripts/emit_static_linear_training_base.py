#!/usr/bin/env python
"""Emit a canonical content-pinned static-linear TrainingRunSpec base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.contracts.spec_storage import training_spec_canonical_bytes, training_spec_sha256
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.runtime.training_run_specs import register_rlrmp_cs_supervised_method
from rlrmp.train.adaptive_epsilon_native import (
    ensure_adaptive_epsilon_training_method_registered,
)
from rlrmp.train.static_linear_native import (
    STATIC_LINEAR_TRAINING_DISTRIBUTIONS,
    author_static_linear_training_base_from_canonical,
)
from rlrmp.train.training_base_routes import route_training_base


def main() -> int:
    """Build, validate, and write one canonical static-linear base."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument(
        "--training-distribution",
        choices=STATIC_LINEAR_TRAINING_DISTRIBUTIONS,
        default="nominal",
    )
    args = parser.parse_args()
    ensure_adaptive_epsilon_training_method_registered()
    register_rlrmp_cs_supervised_method()
    source = json.loads(args.base.read_text(encoding="utf-8"))
    if "feedbax_training_run_spec" in source:
        source = source["feedbax_training_run_spec"]
    base = TrainingRunSpec.model_validate(source)
    authored = author_static_linear_training_base_from_canonical(
        base,
        training_distribution=args.training_distribution,
    )
    authored = route_training_base(authored, issue=args.issue, row_id=args.row_id)
    payload = authored.model_dump(mode="json", exclude_none=True)
    output_bytes = training_spec_canonical_bytes(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)
    print(f"STATIC_LINEAR_BASE path={args.output} sha256={training_spec_sha256(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
