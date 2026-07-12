"""Generated-config entry tests for native guided distillation."""

from __future__ import annotations

import importlib.util

import pytest

from rlrmp.train.distillation_entry import load_distillation_run_spec
from rlrmp.train.distillation_native import guided_kernel
from rlrmp.train.training_configs import GuidedDistillationConfig


def test_guided_typed_config_loads_and_refreshes_tracked_native_spec() -> None:
    config = GuidedDistillationConfig(n_batches=3, batch_size=2, n_replicates=1)
    spec = load_distillation_run_spec(config, method="guided_distillation")

    assert spec["n_train_batches"] == 3
    assert spec["batch_size"] == 2
    assert spec["model_contract"]["n_replicates"] == 1
    assert spec["training_entry"]["module"] == "rlrmp.train.distillation_entry"
    assert spec["teacher_bank"]["materializer"] == (
        "rlrmp.train.distillation_native.guided_kernel.materialize_teacher_batch"
    )
    assert spec["feedbax_training_run_spec"]["method_ref"] == {
        "package": "rlrmp",
        "name": "guided_distillation",
        "version": "v1",
    }


def test_guided_forcing_schedule_remains_method_specific() -> None:
    spec = load_distillation_run_spec(
        GuidedDistillationConfig(),
        method="guided_distillation",
    )
    assert guided_kernel.forcing_fraction_for_batch(spec, 0) == pytest.approx(0.0)
    assert guided_kernel.forcing_fraction_for_batch(spec, 1500) == pytest.approx(0.5)
    assert guided_kernel.forcing_fraction_for_batch(spec, 4000) == pytest.approx(0.9)


def test_retired_guided_public_module_is_absent() -> None:
    assert importlib.util.find_spec("rlrmp.train.guided_distillation") is None
