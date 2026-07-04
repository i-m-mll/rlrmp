"""AST data-lint regression tests (issue ea6ccb4).

Enforces that generated/adopted empirical datasets do not live as module-level
source constants under ``src/``, and proves the lint catches the two named tables
(``DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT`` and ``BROAD_EPSILON_LEVELS``) while
ignoring dimensions, tolerances, and enum-like labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlrmp.data_products.lint import (
    ALLOWLIST,
    scan_source,
    scan_tree,
    significant_figures,
    violations,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# Pre-migration source snippet: the two named generated-data tables as literals.
# The lint MUST flag both. (Negative canary proving the lint has teeth.)
_NAMED_TABLES_SNIPPET = """
DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT = {
    "command_input_pulse": {
        "early": 0.022194585242892,
        "mid": 0.01739291144150351,
        "late": 0.007727047798606057,
    },
    "process_epsilon_velocity_xy": {
        "early": 2.53474407996,
        "mid": 2.05747045529,
        "late": 1.08847767512,
    },
}
BROAD_EPSILON_LEVELS = {
    "moderate": {
        "gamma_factor": 1.4,
        "closed_loop_epsilon_energy_15cm": 1.518885046213267e-06,
        "closed_loop_epsilon_l2_15cm": 0.0012324305441740995,
        "delta_v_percent": 4.041729916548296,
    },
    "strong": {
        "gamma_factor": 1.05,
        "closed_loop_epsilon_energy_15cm": 5.421868381615368e-06,
        "closed_loop_epsilon_l2_15cm": 0.0023284905801002004,
        "delta_v_percent": 7.460371202249536,
    },
}
"""

# Things the lint MUST ignore: a dimension constant, a solver tolerance, an
# enum-like string tuple, and a low-precision convention tuple outside a
# designated science-data module.
_IGNORED_SNIPPET = """
BROAD_EPSILON_DIM = 8
SOLVER_TOLERANCE = 1e-9
PERTURBATION_BINS = ("nominal", "process_epsilon", "command_input", "sensory_feedback")
DEFAULT_AMPLITUDE_FACTORS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
"""

_LOW_PRECISION_CALIBRATION_TABLE_SNIPPET = """
DEFAULT_AMPLITUDE_FACTORS = (
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
)
"""

# A value produced by a loader call is not a literal and is never flagged.
_LOADER_FED_SNIPPET = """
DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT = load_open_loop_calibration().peak_delta_x_per_unit
"""


@pytest.mark.feedbax_contract
def test_data_lint_flags_the_two_named_generated_tables() -> None:
    findings = scan_source(_NAMED_TABLES_SNIPPET, "example/pre_migration.py")
    names = {finding.name for finding in findings}
    assert "DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT" in names
    assert "BROAD_EPSILON_LEVELS" in names
    assert all(finding.n_high_precision > 3 for finding in findings)


@pytest.mark.feedbax_contract
def test_data_lint_ignores_dimensions_tolerances_and_labels() -> None:
    findings = scan_source(_IGNORED_SNIPPET, "example/conventions.py")
    assert findings == [], [f.name for f in findings]


@pytest.mark.feedbax_contract
def test_data_lint_flags_low_precision_calibration_table_shape() -> None:
    findings = scan_source(
        _LOW_PRECISION_CALIBRATION_TABLE_SNIPPET,
        "src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py",
    )
    assert [finding.name for finding in findings] == ["DEFAULT_AMPLITUDE_FACTORS"]
    assert findings[0].n_high_precision == 0
    assert findings[0].n_float_leaves == 14


@pytest.mark.feedbax_contract
def test_data_lint_ignores_loader_fed_assignments() -> None:
    findings = scan_source(_LOADER_FED_SNIPPET, "example/loader_fed.py")
    assert findings == [], [f.name for f in findings]


@pytest.mark.feedbax_contract
def test_src_tree_has_no_unallowlisted_generated_constants() -> None:
    found = violations(SRC_ROOT, repo_root=REPO_ROOT)
    assert found == [], [
        f"{finding.key} (line {finding.lineno}, "
        f"{finding.n_high_precision} high-precision float leaves)"
        for finding in found
    ]


@pytest.mark.feedbax_contract
def test_data_lint_allowlist_entries_exist_and_carry_rationale() -> None:
    # Every allowlisted literal must still exist in the tree (no stale allowlist),
    # and each entry must carry a non-trivial rationale.
    tree_keys = {finding.key for finding in scan_tree(SRC_ROOT, repo_root=REPO_ROOT)}
    for key, rationale in ALLOWLIST.items():
        assert key in tree_keys, f"stale allowlist entry with no matching literal: {key}"
        assert isinstance(rationale, str) and len(rationale.strip()) >= 40, (
            f"allowlist entry {key} lacks a substantive rationale"
        )
    # The one adopted PGD radius-source table is the expected allowlisted case.
    assert "src/rlrmp/train/cs_perturbation_training.py::PGD_SISU_MAX_RADIUS_SOURCES" in ALLOWLIST


@pytest.mark.feedbax_contract
def test_significant_figures_helper() -> None:
    assert significant_figures(1.4) == 2
    assert significant_figures(1.518885046213267e-06) > 6
    assert significant_figures(5.0) == 1
    assert significant_figures(0.05) == 1
