"""Tests for stable machine-JSON output helpers."""

from __future__ import annotations

import json

from rlrmp.io import compact_json_dumps, write_compact_json


def test_compact_json_dumps_is_sorted_single_line() -> None:
    encoded = compact_json_dumps({"b": [2, 1], "a": {"z": True}})

    assert encoded == '{"a":{"z":true},"b":[2,1]}\n'
    assert len(encoded.splitlines()) == 1


def test_write_compact_json_round_trips_without_pretty_indent(tmp_path) -> None:
    path = tmp_path / "payload.json"
    write_compact_json(path, {"outer": {"inner": [1, 2, 3]}})

    text = path.read_text(encoding="utf-8")
    assert "\n  " not in text
    assert json.loads(text) == {"outer": {"inner": [1, 2, 3]}}
