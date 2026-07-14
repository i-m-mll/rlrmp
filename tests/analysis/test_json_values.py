"""Semantic tests for JSON-safe analysis scalar conversion."""

from __future__ import annotations

import math

from rlrmp.analysis.json_values import json_float, json_float_or_none


def test_json_float_preserves_finite_values_and_names_nonfinite_values() -> None:
    assert json_float(1.25) == 1.25
    assert json_float(math.inf) == "inf"
    assert json_float(-math.inf) == "-inf"
    assert json_float(math.nan) == "nan"


def test_json_float_or_none_preserves_absence() -> None:
    assert json_float_or_none(None) is None
    assert json_float_or_none(-2.5) == -2.5
