"""Contract tests for the destination-based data-in-code scanner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlrmp.data_products.data_in_code import (
    DATA_IN_CODE_ALLOWLIST,
    HP_NAME_LEXICON,
    DataInCodePolicyError,
    default_spec_constructor_names,
    policy_for_finding,
    scan_source,
    scan_tree,
    tier_for_relpath,
    validate_findings,
    violations,
    write_baseline,
)


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_negative_canary_flags_argv_row_function() -> None:
    findings = scan_source(
        """
def planned_rows():
    return [["--controller-lr", "0.003", "--n-batches", "12000"]]
""",
        "src/rlrmp/example.py",
    )

    assert _keys(findings) == ["src/rlrmp/example.py::planned_rows::argv_rows"]


def test_argv_row_detector_ignores_argparse_setup() -> None:
    findings = scan_source(
        """
def build_parser(parser):
    parser.add_argument("--controller-lr", default=0.003)
""",
        "scripts/example.py",
    )

    assert findings == []


def test_spec_flow_flags_literal_keywords_into_known_specs() -> None:
    findings = scan_source(
        """
def make_spec():
    return TrainingRunSpec(n_batches=12000, batch_size=256)
""",
        "src/rlrmp/example.py",
        constructor_names={"TrainingRunSpec"},
    )

    assert _keys(findings) == ["src/rlrmp/example.py::make_spec::spec_flow"]


def test_spec_flow_ignores_variable_keywords() -> None:
    findings = scan_source(
        """
def make_spec(n_batches):
    return TrainingRunSpec(n_batches=n_batches)
""",
        "src/rlrmp/example.py",
        constructor_names={"TrainingRunSpec"},
    )

    assert findings == []


def test_default_bundle_flags_function_returned_hp_dict() -> None:
    findings = scan_source(
        """
def _base_hps():
    return {"batch_size": 256, "controller_lr": 0.003, "seed": 1, "label": "run"}
""",
        "src/rlrmp/train/example.py",
    )

    assert _keys(findings) == ["src/rlrmp/train/example.py::_base_hps::default_bundle"]


def test_default_bundle_threshold_ignores_small_dict() -> None:
    findings = scan_source(
        """
def _base_hps():
    return {"batch_size": 256, "controller_lr": 0.003}
""",
        "src/rlrmp/train/example.py",
    )

    assert findings == []


def test_negative_canary_flags_scalar_hp_default() -> None:
    findings = scan_source(
        "DEFAULT_LEARNING_RATE = 3e-3\n",
        "src/rlrmp/train/example.py",
    )

    assert _keys(findings) == ["src/rlrmp/train/example.py::DEFAULT_LEARNING_RATE::hp_constant"]


def test_hp_constant_ignores_dimensions_tolerances_and_labels() -> None:
    findings = scan_source(
        """
BROAD_EPSILON_DIM = 8
_ATOL = 1e-9
PERTURBATION_LABELS = ("nominal", "strong")
""",
        "src/rlrmp/example.py",
    )

    assert findings == []


def test_empirical_table_detector_delegates_existing_lint() -> None:
    findings = scan_source(
        """
DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT = {
    "command_input_pulse": {
        "early": 0.022194585242892,
        "mid": 0.01739291144150351,
        "late": 0.007727047798606057,
        "very_late": 0.007727047798606057,
    },
}
""",
        "src/rlrmp/example.py",
    )

    assert _keys(findings) == [
        "src/rlrmp/example.py::DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT::empirical_table"
    ]


def test_tier_assignment_covers_results_scripts() -> None:
    assert tier_for_relpath("src/rlrmp/example.py") == "src"
    assert tier_for_relpath("scripts/example.py") == "scripts"
    assert tier_for_relpath("results/abc1234/scripts/example.py") == "results_scripts"
    assert tier_for_relpath("tests/example.py") is None


def test_results_script_hp_constants_are_advisory() -> None:
    findings = scan_source(
        "DEFAULT_LEARNING_RATE = 3e-3\n",
        "results/abc1234/scripts/example.py",
    )

    assert _keys(findings) == [
        "results/abc1234/scripts/example.py::DEFAULT_LEARNING_RATE::hp_constant"
    ]
    assert policy_for_finding(findings[0]) == "advisory"


def test_qualname_tracking_covers_nested_function_findings() -> None:
    findings = scan_source(
        """
class Planner:
    def planned(self):
        def inner():
            return [["--seed", "1"]]
        return inner()
""",
        "src/rlrmp/example.py",
    )

    assert _keys(findings) == ["src/rlrmp/example.py::Planner.planned.inner::argv_rows"]


def test_lexicon_is_sorted_and_non_empty() -> None:
    assert HP_NAME_LEXICON
    assert tuple(sorted(HP_NAME_LEXICON)) == HP_NAME_LEXICON


def test_default_spec_constructor_names_include_current_builders() -> None:
    names = default_spec_constructor_names(REPO_ROOT)
    assert "TrainingRunSpec" in names
    assert "build_feedbax_training_run_spec" in names
    assert "build_minimax_training_run_spec" in names
    assert "build_distillation_spec" in names


def test_allowlist_entries_exist_and_have_rationales() -> None:
    findings = {finding.key for finding in scan_tree(REPO_ROOT)}
    for key, rationale in DATA_IN_CODE_ALLOWLIST.items():
        assert key in findings
        assert isinstance(rationale, str) and len(rationale.strip()) >= 40


def test_live_tree_matches_committed_baseline() -> None:
    validate_findings(REPO_ROOT)
    assert violations(REPO_ROOT) == []


def test_live_planned_row_functions_have_been_drained() -> None:
    findings = scan_tree(REPO_ROOT)
    planned_row_findings = [
        finding
        for finding in findings
        if finding.detector == "argv_rows" and "planned_" in finding.qualname
    ]

    assert planned_row_findings == []


def test_write_baseline_refuses_growth(tmp_path: Path) -> None:
    repo_root = tmp_path
    src = repo_root / "src" / "rlrmp"
    src.mkdir(parents=True)
    (repo_root / "ci").mkdir()
    (repo_root / "ci" / "data_in_code_baseline.json").write_text("[]\n", encoding="utf-8")
    (src / "example.py").write_text("DEFAULT_LEARNING_RATE = 3e-3\n", encoding="utf-8")

    with pytest.raises(DataInCodePolicyError, match="refusing to grow"):
        write_baseline(repo_root)


def test_write_baseline_initial_seed(tmp_path: Path) -> None:
    repo_root = tmp_path
    src = repo_root / "src" / "rlrmp"
    src.mkdir(parents=True)
    (src / "example.py").write_text("DEFAULT_LEARNING_RATE = 3e-3\n", encoding="utf-8")

    keys = write_baseline(repo_root)

    assert keys == ["src/rlrmp/example.py::DEFAULT_LEARNING_RATE::hp_constant"]
    assert json.loads((repo_root / "ci" / "data_in_code_baseline.json").read_text()) == keys


def test_write_baseline_excludes_advisory_findings(tmp_path: Path) -> None:
    repo_root = tmp_path
    scripts = repo_root / "results" / "abc1234" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "example.py").write_text("DEFAULT_LEARNING_RATE = 3e-3\n", encoding="utf-8")

    keys = write_baseline(repo_root)

    assert keys == []


def _keys(findings: list) -> list[str]:
    return [finding.key for finding in findings]
