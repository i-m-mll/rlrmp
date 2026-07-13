#!/usr/bin/env python
"""Emit a content-pinned linear-recurrent base from a canonical C&S base."""

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
from rlrmp.train.linear_recurrent_native import (
    LINEAR_RECURRENT_TRAINING_DISTRIBUTIONS,
    author_linear_recurrent_training_base_from_canonical,
)
from rlrmp.train.training_base_routes import route_training_base


def main() -> int:
    """Build, validate, and write one canonical recurrent base."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--issue")
    parser.add_argument("--row-id")
    parser.add_argument(
        "--training-distribution",
        choices=LINEAR_RECURRENT_TRAINING_DISTRIBUTIONS,
        default="nominal",
    )
    args = parser.parse_args()
    ensure_adaptive_epsilon_training_method_registered()
    register_rlrmp_cs_supervised_method()
    source = json.loads(args.base.read_text(encoding="utf-8"))
    if "feedbax_training_run_spec" in source:
        source = source["feedbax_training_run_spec"]
    base = TrainingRunSpec.model_validate(source)
    authored = author_linear_recurrent_training_base_from_canonical(
        base,
        training_distribution=args.training_distribution,
    )
    if bool(args.issue) != bool(args.row_id):
        parser.error("--issue and --row-id must be supplied together")
    if args.issue is not None:
        authored = route_training_base(authored, issue=args.issue, row_id=args.row_id)
    payload = authored.model_dump(mode="json", exclude_none=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(training_spec_canonical_bytes(payload))
    print(f"LINEAR_RECURRENT_BASE path={args.output} sha256={training_spec_sha256(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
