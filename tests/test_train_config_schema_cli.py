"""Training config schema and generated CLI regression tests."""

from __future__ import annotations

import argparse

import pytest

from rlrmp.runtime.params_models import params_model_for
from rlrmp.train.cs_nominal_gru import (
    CS_NOMINAL_GRU_PARAMS_REF,
    CsNominalGruConfig,
    build_parser,
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
        action.dest: action
        for action in parser._actions
        if isinstance(action, argparse.Action)
    }
    assert set(defaults).issubset(action_by_dest)
    assert action_by_dest["plant_backend"].choices == ["cs_lss", "legacy_causal_simplefeedback"]
    assert action_by_dest["loss_objective"].choices == [
        "partial_feedbax_terms",
        "partial_net_output_force_filter",
        "full_analytical_qrf",
    ]


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
