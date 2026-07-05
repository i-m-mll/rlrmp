"""Tests for the scenario-driven packing benchmark harness."""

from __future__ import annotations

from rlrmp.benchmarks.packing import _cs_nominal_gru_pgd_training_config
from rlrmp.train.cs_perturbation_training import PgdFullStateEpsilonTrainingConfig


def test_packing_pgd_defaults_match_training_config_owner() -> None:
    benchmark_config = _cs_nominal_gru_pgd_training_config(None)
    training_config = PgdFullStateEpsilonTrainingConfig()

    assert benchmark_config.n_steps == training_config.n_steps == 3
    assert benchmark_config.step_size_fraction == training_config.step_size_fraction == 0.25
