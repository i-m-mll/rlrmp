"""Tests for fail-closed run-spec metadata accessors."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlrmp.runtime.run_spec_access import (
    RunSpecAccessError,
    require_run_dt,
    require_run_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTED_SEED_FILES = (
    "src/rlrmp/analysis/pipelines/cs_gru_standard_materialization.py",
    "src/rlrmp/analysis/pipelines/gru_broad_epsilon_attribution.py",
    "src/rlrmp/analysis/pipelines/gru_checkpoint_selection.py",
    "src/rlrmp/analysis/pipelines/gru_evaluation_diagnostics.py",
    "src/rlrmp/analysis/pipelines/gru_feedback_ablation.py",
    "src/rlrmp/analysis/pipelines/gru_perturbation_bank.py",
    "src/rlrmp/analysis/pipelines/gru_pilot_figures.py",
    "src/rlrmp/analysis/pipelines/gru_steady_state_perturbation_bank.py",
    "src/rlrmp/analysis/pipelines/gru_worst_case_epsilon_audit.py",
    "src/rlrmp/analysis/pipelines/objective_comparator.py",
    "src/rlrmp/analysis/pipelines/sisu_spectrum_diagnostics.py",
    "src/rlrmp/model/cs_lss_gru.py",
    "src/rlrmp/model/feedbax_graph.py",
    "src/rlrmp/train/closed_loop_distillation.py",
)
SEED_LITERAL_FALLBACK = re.compile(r"\.get\(\s*['\"]seed['\"]\s*,\s*(?:0|42)\s*\)")


def test_require_run_seed_returns_present_value() -> None:
    assert require_run_seed({"seed": "7"}) == 7


def test_require_run_seed_names_missing_spec_source() -> None:
    with pytest.raises(RunSpecAccessError) as exc_info:
        require_run_seed({}, source="results/abc1234/runs/demo.json")

    assert exc_info.value.field == "seed"
    assert exc_info.value.source == "results/abc1234/runs/demo.json"
    assert "results/abc1234/runs/demo.json" in str(exc_info.value)
    assert "seed" in str(exc_info.value)


def test_require_run_dt_prefers_game_card_value() -> None:
    hps = SimpleNamespace(dt=0.02)
    assert require_run_dt({"game_card": {"dt": "0.01"}}, hps) == pytest.approx(0.01)


def test_require_run_dt_uses_hps_when_game_card_lacks_dt() -> None:
    hps = SimpleNamespace(dt=0.02)
    assert require_run_dt({"game_card": {}}, hps) == pytest.approx(0.02)


def test_require_run_dt_names_missing_spec_source() -> None:
    with pytest.raises(RunSpecAccessError) as exc_info:
        require_run_dt({"game_card": {}}, SimpleNamespace(), source=Path("runs/missing.json"))

    assert exc_info.value.field == "game_card.dt"
    assert exc_info.value.source == "runs/missing.json"
    assert "runs/missing.json" in str(exc_info.value)
    assert "dt is plant physics" in str(exc_info.value)


def test_converted_files_have_no_literal_seed_get_fallbacks() -> None:
    offenders = []
    for path in CONVERTED_SEED_FILES:
        source = (REPO_ROOT / path).read_text(encoding="utf-8")
        if SEED_LITERAL_FALLBACK.search(source):
            offenders.append(path)

    assert offenders == []
