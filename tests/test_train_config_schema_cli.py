"""Training config schema and generated CLI regression tests."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from rlrmp.runtime.params_models import params_model_for
from rlrmp.train.cs_nominal_gru import (
    CS_NOMINAL_GRU_PARAMS_REF,
    CsNominalGruConfig,
    build_parser,
)
from rlrmp.train.run_spec_authoring import build_parser as authoring_build_parser
from rlrmp.train.distillation_native import closed_loop as closed_loop_distillation
from rlrmp.train.distillation_native import guided as guided_distillation
from rlrmp.train.minimax import MinimaxConfig
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


def test_generated_parser_tracks_config_defaults_and_choices() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    defaults = CsNominalGruConfig().model_dump(mode="python")

    for name, expected in defaults.items():
        assert getattr(args, name) == expected

    action_by_dest = {
        action.dest: action for action in parser._actions if isinstance(action, argparse.Action)
    }
    assert set(defaults).issubset(action_by_dest)
    assert action_by_dest["plant_backend"].choices == ["cs_lss", "legacy_causal_simplefeedback"]
    assert action_by_dest["loss_objective"].choices == [
        "partial_feedbax_terms",
        "partial_net_output_force_filter",
        "full_analytical_qrf",
    ]


def test_generated_parser_is_owned_by_run_spec_authoring() -> None:
    assert build_parser is authoring_build_parser
    assert build_parser.__module__ == "rlrmp.train.run_spec_authoring"

    nominal_source = REPO_ROOT / "src/rlrmp/train/cs_nominal_gru.py"
    tree = ast.parse(nominal_source.read_text(encoding="utf-8"))
    imports_by_module = {
        node.module: {alias.name for alias in node.names}
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert "build_parser" in imports_by_module["rlrmp.train.run_spec_authoring"]
    assert "build_parser" not in imports_by_module["rlrmp.train.executor.cs_supervised"]


def test_generated_parser_validates_with_config_model() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--n-train-batches",
            "7",
            "--batch-size",
            "3",
            "--no-quiet-progress",
            "--plant-backend",
            "cs_lss",
        ]
    )

    config = CsNominalGruConfig.model_validate(vars(args))

    assert config.n_train_batches == 7
    assert config.batch_size == 3
    assert config.quiet_progress is False


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

    guided_args = guided_distillation.build_parser().parse_args([])
    closed_args = closed_loop_distillation._build_parser().parse_args([])
    assert vars(guided_args) == GuidedDistillationConfig().model_dump(mode="python")
    assert vars(closed_args) == ClosedLoopDistillationConfig().model_dump(mode="python")


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
    path = REPO_ROOT / "src/rlrmp/train/minimax_native/authoring.py"
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
