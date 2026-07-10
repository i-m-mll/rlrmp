"""Materialize delayed SISU=1 vs SISU=0 fixed-bank velocity profiles."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from typing import Any

from rlrmp.paths import REPO_ROOT


DRIVER_SPEC_PATH = (
    REPO_ROOT / "results" / "7c1f7ed" / "runs" / "delayed_sisu_velocity_profiles.json"
)
DRIVER_SPEC_SCHEMA_ID = "rlrmp.delayed_sisu_velocity_profiles.driver_spec"
DRIVER_SPEC_SCHEMA_VERSION = "rlrmp.delayed_sisu_velocity_profiles.driver_spec.v1"
DRIVER_SPEC_CONTENT_SHA256 = "69bd5ac3f9c39760457cf5d716c842a6f21e0c9bb275ad8dad24776dbda434c8"


def load_driver_spec() -> dict[str, Any]:
    """Load and validate the exact tracked historical materialization spec."""

    payload = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("delayed SISU driver spec must be a JSON object")
    if payload.get("schema_id") != DRIVER_SPEC_SCHEMA_ID:
        raise ValueError(f"unsupported delayed SISU driver schema_id: {payload.get('schema_id')!r}")
    if payload.get("schema_version") != DRIVER_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported delayed SISU driver schema_version: {payload.get('schema_version')!r}"
        )
    stored_hash = payload.get("content_sha256")
    semantic_payload = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed_hash = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stored_hash != DRIVER_SPEC_CONTENT_SHA256 or computed_hash != DRIVER_SPEC_CONTENT_SHA256:
        raise ValueError(
            "delayed SISU driver spec content identity mismatch: "
            f"stored={stored_hash!r}, computed={computed_hash!r}, "
            f"expected={DRIVER_SPEC_CONTENT_SHA256!r}"
        )
    if not payload.get("run_refs") or not payload.get("sisu_levels"):
        raise ValueError("delayed SISU driver spec requires run_refs and sisu_levels")
    return payload


def main() -> None:
    """Run the delayed fixed-bank velocity materializer with 7c1f7ed defaults."""

    script = (
        REPO_ROOT
        / "results"
        / "40e1911"
        / "scripts"
        / "materialize_delayed_timing_hold_lane_velocity_profiles.py"
    )
    if len(sys.argv) == 1:
        spec = load_driver_spec()
        sys.argv.extend(["--result-experiment", spec["result_experiment"]])
        sys.argv.extend(["--topic", spec["topic"]])
        for level in spec["sisu_levels"]:
            sys.argv.extend(["--sisu-level", str(level)])
        for run_ref in spec["run_refs"]:
            sys.argv.extend(["--run-ref", run_ref])
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
