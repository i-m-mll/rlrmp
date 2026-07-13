"""Registered training-science lowering contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.train.config_materialization import build_hps
from rlrmp.train import science_lowering
from rlrmp.train.science_lowering import (
    MODE_LOWERERS,
    lower_training_fidelity,
    lower_training_mode,
    lower_training_science,
)
from rlrmp.train.science_vocabulary import ScienceMode
from rlrmp.train.training_configs import CsNominalGruConfig


def _hps(**overrides: object):
    values = CsNominalGruConfig(
        issue="test",
        output_dir="_artifacts/test/runs/test",
    ).model_dump(mode="python")
    values.update(overrides)
    return build_hps(values)


@pytest.mark.parametrize(
    ("overrides", "expected_mode", "expected_capabilities"),
    [
        ({}, ScienceMode.NOMINAL, ()),
        (
            {"perturbation_training": True},
            ScienceMode.PERTURBATION,
            ("perturbation",),
        ),
        (
            {"target_relative_multitarget": True},
            ScienceMode.TARGET_RELATIVE,
            ("target_relative",),
        ),
        (
            {
                "target_relative_multitarget": True,
                "initial_hidden_encoder": True,
                "perturbation_training": True,
            },
            f"{ScienceMode.TARGET_RELATIVE_H0}+{ScienceMode.PERTURBATION}",
            ("target_relative", "perturbation"),
        ),
        (
            {
                "target_relative_multitarget": True,
                "broad_epsilon_training": True,
            },
            f"{ScienceMode.TARGET_RELATIVE}+{ScienceMode.BROAD_EPSILON}",
            ("target_relative", "broad_epsilon"),
        ),
        (
            {
                "target_relative_multitarget": True,
                "broad_epsilon_pgd_training": True,
            },
            f"{ScienceMode.TARGET_RELATIVE}+{ScienceMode.BROAD_EPSILON_PGD}",
            ("target_relative", "broad_epsilon_pgd"),
        ),
        (
            {
                "target_relative_multitarget": True,
                "policy_adversary_training": True,
                "policy_adversary_radius_15cm": 0.001,
                "policy_adversary_radius_source": "effective_020a65b_pgd_training_radius",
            },
            f"{ScienceMode.TARGET_RELATIVE}+{ScienceMode.POLICY_ADVERSARY}",
            ("target_relative", "policy_adversary"),
        ),
        (
            {
                "delayed_reach": True,
                "target_relative_multitarget": True,
                "broad_epsilon_pgd_training": True,
            },
            (
                f"{ScienceMode.DELAYED_REACH}+{ScienceMode.TARGET_RELATIVE}+"
                f"{ScienceMode.BROAD_EPSILON_PGD}+{ScienceMode.PERTURBATION}"
            ),
            ("delayed_reach", "target_relative", "broad_epsilon_pgd", "perturbation"),
        ),
    ],
)
def test_registered_capability_lowerers_compose_in_spec_order(
    overrides: dict[str, object],
    expected_mode: str,
    expected_capabilities: tuple[str, ...],
) -> None:
    lowered = lower_training_science(_hps(**overrides))

    assert lowered.training_mode == expected_mode
    assert lowered.lowerer_ids[:-1] == expected_capabilities
    assert lowered.lowerer_ids[-1] == "objective.partial"


@pytest.mark.parametrize(
    ("objective", "lowerer_id", "implemented_control"),
    [
        ("partial_feedbax_terms", "objective.partial", "nn_output"),
        (
            CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
            "objective.partial_force_filter",
            "nn_output",
        ),
        (
            CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            "objective.full_qrf",
            "net.output^T R_t net.output",
        ),
    ],
)
def test_registered_objective_lowerers_own_loss_and_fidelity_together(
    objective: str,
    lowerer_id: str,
    implemented_control: str,
) -> None:
    lowered = lower_training_science(_hps(loss_objective=objective))

    assert lowered.lowerer_ids == (lowerer_id,)
    assert lowered.loss_spec["objective_profile"] == objective
    assert lowered.fidelity_status["loss_objective"] == objective
    active = lowered.loss_spec["active_cs_terms"]
    control = active.get("control", active.get("control_r"))
    assert control["term"] == implemented_control


def test_science_lowerer_registry_records_capability_owners() -> None:
    registrations = {item.lowerer_id: item for item in MODE_LOWERERS.registrations()}
    owners = {lowerer_id: item.owner for lowerer_id, item in registrations.items()}

    assert owners["perturbation"] == "rlrmp.train.fixed_target_perturbation_training"
    assert owners["broad_epsilon"] == "rlrmp.train.broad_epsilon_training"
    assert owners["policy_adversary"] == "rlrmp.train.policy_adversary_native"
    assert owners["target_relative"] == "rlrmp.train.cs_perturbation_training"
    assert owners["delayed_reach"] == "rlrmp.train.delayed_reach"
    assert {
        lowerer_id: item.lowerer.__module__ for lowerer_id, item in registrations.items()
    } == owners


def test_mode_and_fidelity_only_lowering_never_build_full_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called():
        raise AssertionError("full analytical game construction is loss-only")

    monkeypatch.setattr(science_lowering, "build_canonical_game", fail_if_called)
    hps = _hps(loss_objective=CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE)

    assert lower_training_mode(hps).training_mode == ScienceMode.NOMINAL
    assert lower_training_fidelity(hps)["loss_objective"] == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE


def test_retired_switch_functions_and_mode_constant_family_cannot_return() -> None:
    authoring = ast.parse(Path("src/rlrmp/train/run_spec_authoring.py").read_text(encoding="utf-8"))
    function_names = {
        node.name for node in ast.walk(authoring) if isinstance(node, ast.FunctionDef)
    }
    assert {"_training_mode", "_loss_spec", "_fidelity_status"}.isdisjoint(function_names)

    forbidden_assignments: list[tuple[str, str]] = []
    for path in Path("src/rlrmp").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and "TRAINING_MODE" in target.id:
                    forbidden_assignments.append((str(path), target.id))
    assert forbidden_assignments == []


def test_distillation_stays_outside_nominal_gru_science_modes() -> None:
    assert all("distillation" not in mode.value for mode in ScienceMode)
