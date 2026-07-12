"""Consumed data-product identity envelope assembly tests."""

from __future__ import annotations

from types import SimpleNamespace


def _product(identity_hash: str) -> SimpleNamespace:
    return SimpleNamespace(product_identity_hash=identity_hash)


def test_train_consumed_calibration_budget_identities_shape(monkeypatch) -> None:
    from rlrmp.train import cs_perturbation_training as training

    monkeypatch.setattr(
        training,
        "load_open_loop_calibration",
        lambda: _product("sha256:unit-calibration"),
    )
    monkeypatch.setattr(
        training,
        "load_broad_epsilon_anchors",
        lambda: _product("sha256:unit-broad-epsilon"),
    )

    assert training.consumed_calibration_budget_identities(
        calibration_consumed=True,
        broad_epsilon_consumed=True,
    ) == [
        {
            "role": "perturbation_open_loop_calibration",
            "schema": "rlrmp.perturbation_open_loop_calibration.v2",
            "hash": "sha256:unit-calibration",
        },
        {
            "role": "broad_epsilon_budget_anchors",
            "schema": "rlrmp.broad_epsilon_budget_anchors.v1",
            "hash": "sha256:unit-broad-epsilon",
        },
    ]
    assert (
        training.consumed_calibration_budget_identities(
            calibration_consumed=False,
            broad_epsilon_consumed=False,
        )
        == []
    )


def test_eval_consumed_calibration_identity_shape(monkeypatch) -> None:
    from rlrmp.data_products import calibration as calibration_products
    from rlrmp.eval import recipes

    monkeypatch.setattr(
        calibration_products,
        "load_open_loop_calibration",
        lambda: _product("sha256:unit-calibration"),
    )

    params = recipes._with_eval_consumed_calibration_identity({"bank_mode": "calibrated"})

    assert params["consumed_data_identities"] == [
        {
            "role": "perturbation_open_loop_calibration",
            "schema": "rlrmp.perturbation_open_loop_calibration.v2",
            "hash": "sha256:unit-calibration",
        }
    ]
    assert recipes._with_eval_consumed_calibration_identity({"bank_mode": "raw"}) == {
        "bank_mode": "raw"
    }


def test_calibration_computation_consumed_defaults_identity_shape(monkeypatch) -> None:
    from rlrmp.data_products import calibration_computation as calibration

    monkeypatch.setattr(
        calibration,
        "load_perturbation_calibration_defaults",
        lambda: _product("sha256:unit-defaults"),
    )

    assert calibration._consumed_default_identities() == [
        {
            "role": "perturbation_calibration_defaults",
            "schema": "rlrmp.perturbation_calibration_defaults.v1",
            "hash": "sha256:unit-defaults",
        }
    ]
