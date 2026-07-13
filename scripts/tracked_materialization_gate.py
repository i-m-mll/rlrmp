#!/usr/bin/env python3
"""Find tracked expanded/resolved experiment-spec materializations."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_JSON_BYTES = 256 * 1024
MIN_EXPANDED_ENVELOPE_BYTES = 16 * 1024
RUN_ENVELOPE_KEYS = frozenset(
    {
        "checkpointing",
        "feedbax_graph",
        "hps",
        "n_train_batches",
        "training_summary",
    }
)
SPEC_CONTAINER_KEYS = frozenset({"override", "overrides", "spec"})
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\\\/]")
JSON_POINTER = re.compile(r"(?:/(?:[^~/]|~[01])*)*")
GOVERNED_JSON_POINTER_SUFFIXES = (
    ("worker_execution", "effective_phase", "consistency_predicate", "rules", "path"),
    ("worker_execution", "method_contract", "objective_reducers", "path"),
    ("worker_execution", "method_contract", "worker_reducers", "path"),
)


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _objects(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _is_absolute_filesystem_path(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or WINDOWS_ABSOLUTE_PATH.match(value) is not None


def _is_governed_json_pointer(value: str, key_path: tuple[str, ...]) -> bool:
    return JSON_POINTER.fullmatch(value) is not None and any(
        key_path[-len(suffix) :] == suffix for suffix in GOVERNED_JSON_POINTER_SUFFIXES
    )


def _contains_absolute_spec_path(
    value: Any,
    *,
    in_spec: bool = False,
    key_path: tuple[str, ...] = (),
) -> bool:
    if isinstance(value, str):
        return (
            in_spec
            and _is_absolute_filesystem_path(value)
            and not _is_governed_json_pointer(value, key_path)
        )
    if isinstance(value, list):
        return any(
            _contains_absolute_spec_path(item, in_spec=in_spec, key_path=key_path)
            for item in value
        )
    if not isinstance(value, dict):
        return False

    schema_id = value.get("schema_id")
    schema_is_spec = isinstance(schema_id, str) and (
        schema_id.startswith("feedbax.spec.") or ".spec." in schema_id
    )
    for key, item in value.items():
        key_is_spec = key in SPEC_CONTAINER_KEYS or key.endswith("_spec")
        if _contains_absolute_spec_path(
            item,
            in_spec=in_spec or schema_is_spec or key_is_spec,
            key_path=(*key_path, key),
        ):
            return True
    return False


def materialization_reasons(payload: Any, *, byte_size: int | None = None) -> list[str]:
    """Return semantic reasons that ``payload`` is an expanded spec."""
    reasons: list[str] = []

    objects = tuple(_objects(payload))
    if any(
        (
            isinstance(obj.get("feedbax_training_run_spec"), dict)
            and _json_bytes(obj["feedbax_training_run_spec"]) >= MIN_EXPANDED_ENVELOPE_BYTES
        )
        or (
            isinstance(obj.get("rlrmp_run_spec"), dict)
            and _json_bytes(obj["rlrmp_run_spec"]) >= MIN_EXPANDED_ENVELOPE_BYTES
        )
        or (
            isinstance(obj.get("hps"), dict)
            and _json_bytes(obj["hps"]) >= MIN_EXPANDED_ENVELOPE_BYTES
            and len(RUN_ENVELOPE_KEYS.intersection(obj)) >= 3
        )
        for obj in objects
    ):
        reasons.append("expanded_run_envelope")

    # Standalone ``feedbax.spec.graph`` documents are compact authored graph
    # inputs. Only embedding their full payload under an inline graph/base
    # container is a tracked materialization.
    if any(
        isinstance(obj.get(container), dict)
        and "inline" in obj[container]
        and _json_bytes(obj[container]["inline"]) >= MIN_EXPANDED_ENVELOPE_BYTES
        for obj in objects
        for container in ("base", "graph", "feedbax_graph")
    ):
        reasons.append("expanded_inline_envelope")

    payload_bytes = _json_bytes(payload) if byte_size is None else byte_size
    if payload_bytes > MAX_TRACKED_JSON_BYTES:
        reasons.append("oversized_json_payload")

    if _contains_absolute_spec_path(payload):
        reasons.append("absolute_filesystem_path_in_spec")
    return reasons


def scan_tracked(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    listed = subprocess.run(
        ["git", "ls-files", "results/**/*.json"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    found: dict[str, list[str]] = {}
    for relpath in listed:
        try:
            raw = (repo_root / relpath).read_bytes()
            payload = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # JSON validity is enforced independently; this gate does not
            # misclassify unreadable files as materialization findings.
            continue
        reasons = materialization_reasons(payload, byte_size=len(raw))
        if reasons:
            found[relpath] = reasons
    return found
