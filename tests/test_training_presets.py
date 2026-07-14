"""Contract tests for hash-verified training authoring presets."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic_core import PydanticUndefined

from rlrmp.data_products.data_in_code import scan_tree
from rlrmp.train.training_configs import (
    AMPLITUDE_LEVELS,
    BROAD_EPSILON_REFERENCE_REACH_M,
    BroadFullStateEpsilonTrainingConfig,
    ClosedLoopDistillationConfig,
    CsNominalGruConfig,
    FixedTargetPerturbationTrainingConfig,
    GuidedDistillationConfig,
    MinimaxConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
)
from rlrmp.train.training_presets import load_training_presets
import rlrmp.train.training_presets as training_presets_module


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIGS = (
    BroadFullStateEpsilonTrainingConfig,
    ClosedLoopDistillationConfig,
    CsNominalGruConfig,
    FixedTargetPerturbationTrainingConfig,
    GuidedDistillationConfig,
    MinimaxConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
)


def test_training_config_defaults_are_preset_backed_field_by_field() -> None:
    presets = load_training_presets()

    for config_type in TRAINING_CONFIGS:
        defaults = {
            name: field.default
            for name, field in config_type.model_fields.items()
            if field.default is not PydanticUndefined
        }
        expected = presets[config_type.__name__]
        assert set(defaults) == set(expected), config_type.__name__
        for name, value in defaults.items():
            expected_value = expected[name]
            if isinstance(value, tuple):
                expected_value = tuple(expected_value)
            assert value == expected_value, f"{config_type.__name__}.{name}"


def test_training_config_classes_need_no_whole_bundle_exemptions() -> None:
    class_names = {config_type.__name__ for config_type in TRAINING_CONFIGS}
    findings = [
        finding
        for finding in scan_tree(REPO_ROOT)
        if finding.relpath == "src/rlrmp/train/training_configs.py"
        and finding.detector == "default_bundle"
        and any(class_name in finding.qualname for class_name in class_names)
    ]
    assert findings == []


def test_shared_numeric_training_defaults_are_preset_backed() -> None:
    shared = load_training_presets()["shared"]
    assert AMPLITUDE_LEVELS == tuple(shared["amplitude_levels"])
    assert BROAD_EPSILON_REFERENCE_REACH_M == shared["broad_epsilon_reference_reach_m"]


def test_training_presets_have_one_tracked_authority() -> None:
    product = json.loads(
        (
            REPO_ROOT
            / "results/ea6ccb4/data_products/broad_epsilon_budget_anchors.json"
        ).read_text(encoding="utf-8")
    )
    extraction = json.loads(
        (
            REPO_ROOT
            / "results/ea6ccb4/data_products/broad_epsilon_budget_anchors.extraction.json"
        ).read_text(encoding="utf-8")
    )

    assert "training_authoring_presets" not in product["parameters"]
    assert "training_authoring_presets" not in extraction["static_parameters"]
    training_config_source = (
        REPO_ROOT / "src/rlrmp/train/training_configs.py"
    ).read_text(encoding="utf-8")
    assert "load_training_authoring_presets" not in training_config_source


def test_training_preset_loader_fails_closed_on_hash_drift(
    tmp_path: Path, monkeypatch,
) -> None:
    document = json.loads(training_presets_module.TRAINING_PRESET_PATH.read_text())
    document["presets"]["shared"]["broad_epsilon_reference_reach_m"] = 0.2
    drifted = tmp_path / "defaults.json"
    drifted.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(training_presets_module, "TRAINING_PRESET_PATH", drifted)
    load_training_presets.cache_clear()
    try:
        with pytest.raises(ValueError, match="content hash mismatch"):
            load_training_presets()
    finally:
        load_training_presets.cache_clear()


def test_training_config_field_default_dispositions_are_explicit() -> None:
    """Every field is required, preset-backed, or a structural factory."""

    path = REPO_ROOT / "src/rlrmp/train/training_configs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_names = {config_type.__name__ for config_type in TRAINING_CONFIGS}
    dispositions: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in class_names:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(
                statement.target, ast.Name
            ):
                continue
            key = f"{node.name}.{statement.target.id}"
            if statement.value is None:
                dispositions[key] = "required"
            elif _calls_name(statement.value, "training_preset_value"):
                dispositions[key] = "preset_backed"
            elif _has_default_factory(statement.value):
                dispositions[key] = "structural_factory"
            else:
                dispositions[key] = "unclassified"

    assert dispositions
    assert "unclassified" not in dispositions.values(), dispositions


def _calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
        for child in ast.walk(node)
    )


def _has_default_factory(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and any(
        keyword.arg == "default_factory" for keyword in node.keywords
    )
