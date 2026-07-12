"""Training-config schema and launch-interface structural gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rlrmp.runtime.params_models import params_model_for
from rlrmp.train.cs_nominal_gru import (
    CS_NOMINAL_GRU_PARAMS_REF,
    CsNominalGruConfig,
)
from rlrmp.train.minimax_native import MinimaxConfig
from rlrmp.train.training_configs import (
    CLOSED_LOOP_DISTILLATION_PARAMS_REF,
    GUIDED_DISTILLATION_PARAMS_REF,
    MINIMAX_PARAMS_REF,
    ClosedLoopDistillationConfig,
    GuidedDistillationConfig,
)
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_PARAMS_REF,
    FIXED_TARGET_PERTURBATION_PARAMS_REF,
    POLICY_ADVERSARY_PARAMS_REF,
    TARGET_RELATIVE_MULTITARGET_PARAMS_REF,
    FixedTargetPerturbationTrainingConfig,
    PgdFullStateEpsilonTrainingConfig,
    PolicyFullStateEpsilonTrainingConfig,
    TargetRelativeMultiTargetTrainingConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_config_models_are_registered() -> None:
    assert params_model_for(CS_NOMINAL_GRU_PARAMS_REF) is CsNominalGruConfig
    assert (
        params_model_for(FIXED_TARGET_PERTURBATION_PARAMS_REF)
        is FixedTargetPerturbationTrainingConfig
    )
    assert params_model_for(BROAD_EPSILON_PGD_PARAMS_REF) is PgdFullStateEpsilonTrainingConfig
    assert params_model_for(POLICY_ADVERSARY_PARAMS_REF) is PolicyFullStateEpsilonTrainingConfig
    assert (
        params_model_for(TARGET_RELATIVE_MULTITARGET_PARAMS_REF)
        is TargetRelativeMultiTargetTrainingConfig
    )


def test_reflection_generated_training_config_flag_surfaces_are_forbidden() -> None:
    """Scientific config models must not be reflected into argparse flags."""

    offenders: list[str] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for function_node in (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                loaded_attrs = {
                    node.attr
                    for node in ast.walk(function_node)
                    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                }
                calls_add_argument = any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    for node in ast.walk(function_node)
                )
                if "model_fields" in loaded_attrs and calls_add_argument:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{function_node.lineno}:"
                        f"{function_node.name}"
                    )
    assert offenders == []


def test_perturbation_configs_are_strict_pydantic_models() -> None:
    with pytest.raises(ValueError):
        FixedTargetPerturbationTrainingConfig(nominal_fraction=0.2)

    with pytest.raises(ValueError):
        PgdFullStateEpsilonTrainingConfig(enabled=True, n_steps=0)

    with pytest.raises(ValueError):
        PolicyFullStateEpsilonTrainingConfig(enabled=True)


def test_training_config_family_has_one_definition_module() -> None:
    config_classes = (
        CsNominalGruConfig,
        FixedTargetPerturbationTrainingConfig,
        PgdFullStateEpsilonTrainingConfig,
        PolicyFullStateEpsilonTrainingConfig,
        TargetRelativeMultiTargetTrainingConfig,
    )
    assert {config_type.__module__ for config_type in config_classes} == {
        "rlrmp.train.training_configs"
    }

    definitions: list[tuple[str, str]] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions.extend(
            (path.relative_to(REPO_ROOT).as_posix(), node.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.endswith("TrainingConfig")
        )
    assert definitions
    assert {path for path, _name in definitions} == {"src/rlrmp/train/training_configs.py"}


def test_native_trainer_configs_share_unified_definition_module() -> None:
    config_classes = (
        MinimaxConfig,
        GuidedDistillationConfig,
        ClosedLoopDistillationConfig,
    )
    assert {config_type.__module__ for config_type in config_classes} == {
        "rlrmp.train.training_configs"
    }
    assert params_model_for(MINIMAX_PARAMS_REF) is MinimaxConfig
    assert params_model_for(GUIDED_DISTILLATION_PARAMS_REF) is GuidedDistillationConfig
    assert params_model_for(CLOSED_LOOP_DISTILLATION_PARAMS_REF) is ClosedLoopDistillationConfig

    assert GuidedDistillationConfig.model_validate({}) == GuidedDistillationConfig()
    assert ClosedLoopDistillationConfig.model_validate({}) == ClosedLoopDistillationConfig()


def test_legacy_training_config_translator_definitions_are_retired() -> None:
    offenders: list[str] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend(
            f"{path.relative_to(REPO_ROOT)}:{node.name}"
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("config_from_")
            and node.name.endswith("_hps")
        )
    assert offenders == []


def test_minimax_hps_entrypoint_is_a_thin_validated_constructor() -> None:
    path = REPO_ROOT / "src/rlrmp/train/minimax_native/method.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    build_hps = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_hps"
    )
    assert build_hps.end_lineno - build_hps.lineno + 1 <= 10
    loaded_names = {
        node.id
        for node in ast.walk(build_hps)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert {"MinimaxConfig", "_build_hps_from_config"}.issubset(loaded_names)


def test_nominal_checkpoint_plumbing_is_slot_and_barrier_driven() -> None:
    checkpoint_source = (REPO_ROOT / "src/rlrmp/train/executor/checkpoints.py").read_text(
        encoding="utf-8"
    )
    nominal_source = (REPO_ROOT / "src/rlrmp/train/cs_nominal_gru.py").read_text(encoding="utf-8")
    assert "CheckpointSlotSpec" in checkpoint_source
    assert "CheckpointBarrierSpec" in checkpoint_source
    assert "_checkpoint_barrier_from_run_spec" in checkpoint_source
    assert "from rlrmp.train.executor.checkpoints import" in nominal_source


def test_cs_training_entry_modules_remain_below_split_ceiling() -> None:
    for relative_path in (
        "src/rlrmp/train/cs_nominal_gru.py",
        "src/rlrmp/train/cs_perturbation_training.py",
    ):
        line_count = len((REPO_ROOT / relative_path).read_text(encoding="utf-8").splitlines())
        assert line_count < 2_000, f"{relative_path} grew to {line_count} lines"
