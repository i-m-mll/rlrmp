from __future__ import annotations

import csv
from io import StringIO

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax import TaskTrialSpec, WhereDict
from feedbax.objectives.loss import AbstractLoss, TermTree
from feedbax.config.namespace import TreeNamespace

from rlrmp.analysis.pipelines.gru_broad_epsilon_attribution import (
    active_vs_zero_semantics,
    epsilon_summary,
    gradient_pair_metrics,
    loss_delta_summary,
    paired_broad_epsilon_training_specs,
    render_summary_csv,
    summarize_loss_tree,
    truncate_trial_specs,
    zero_epsilon_trial_specs,
)
from rlrmp.train.cs_perturbation_training import (
    BroadFullStateEpsilonTrainingConfig,
    BroadFullStateEpsilonTrainingTaskAdapter,
    FixedTargetPerturbationTrainingTaskAdapter,
)


def _trial_specs() -> TaskTrialSpec:
    return TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.zeros((3, 48), dtype=jnp.float32)}),
        inputs={
            "epsilon": jnp.arange(3 * 4 * 8, dtype=jnp.float32).reshape(3, 4, 8),
            "graph_channel": jnp.ones((3, 4, 2), dtype=jnp.float32),
            "effector_target": TreeNamespace(pos=jnp.zeros((3, 4, 2), dtype=jnp.float32)),
        },
        targets={
            "mechanics.effector.pos": TreeNamespace(
                value=jnp.broadcast_to(
                    jnp.asarray([0.15, 0.0], dtype=jnp.float32),
                    (3, 4, 2),
                )
            )
        },
        intervene={},
        extra={"kept": True},
        timeline=TreeNamespace(n_steps=4, epoch_bounds=jnp.asarray([0, 4], dtype=jnp.int32)),
    )


class _ToyBaseTask:
    loss_func: AbstractLoss

    def __init__(self) -> None:
        self.loss_func = _ToyLoss()

    def get_train_trial_with_intervenor_params(self, key, batch_info=None):
        del key, batch_info
        specs = _trial_specs()
        return TaskTrialSpec(
            inits=WhereDict(
                {"mechanics.vector": specs.inits["mechanics.vector"][0]}
            ),
            inputs={
                "epsilon": jnp.zeros((4, 8), dtype=jnp.float32),
                "graph_channel": specs.inputs["graph_channel"][0],
                "effector_target": TreeNamespace(pos=specs.inputs["effector_target"].pos[0]),
            },
            targets={
                "mechanics.effector.pos": TreeNamespace(
                    value=specs.targets["mechanics.effector.pos"].value[0]
                )
            },
            intervene=specs.intervene,
            extra=specs.extra,
            timeline=specs.timeline,
        )


class _ToyLoss(AbstractLoss):
    label: str = "toy"

    def term(self, states, trial_specs, model):
        del states, trial_specs, model
        return jnp.asarray(0.0)


def test_paired_broad_epsilon_training_specs_preserves_outer_perturbation() -> None:
    broad_cfg = TreeNamespace(
        **BroadFullStateEpsilonTrainingConfig(enabled=True, level="moderate").to_hps_dict()
    )
    pert_cfg = TreeNamespace(
        enabled=False,
        nominal_fraction=0.45,
        single_fraction=0.45,
        combined_fraction=0.10,
        calibrated_timing=False,
    )
    task = FixedTargetPerturbationTrainingTaskAdapter(
        BroadFullStateEpsilonTrainingTaskAdapter(_ToyBaseTask(), broad_cfg),
        pert_cfg,
    )

    active, without_broad = paired_broad_epsilon_training_specs(
        task,
        key=jr.PRNGKey(0),
        n_trials=2,
    )

    assert active.inputs["epsilon"].shape == (2, 4, 8)
    assert active.timeline.epoch_bounds.shape == (2, 2)
    np.testing.assert_allclose(without_broad.inputs["epsilon"], 0.0)
    assert not np.allclose(active.inputs["epsilon"], without_broad.inputs["epsilon"])


def test_zero_epsilon_trial_specs_only_replaces_epsilon() -> None:
    trial_specs = _trial_specs()

    zeroed = zero_epsilon_trial_specs(trial_specs)

    np.testing.assert_allclose(zeroed.inputs["epsilon"], 0.0)
    np.testing.assert_allclose(zeroed.inputs["graph_channel"], trial_specs.inputs["graph_channel"])
    np.testing.assert_allclose(
        zeroed.inits["mechanics.vector"], trial_specs.inits["mechanics.vector"]
    )
    assert zeroed.extra == {"kept": True}
    assert active_vs_zero_semantics()["delta_sign"] == "active_minus_zero"


def test_truncate_trial_specs_keeps_consistent_batch_prefix() -> None:
    truncated = truncate_trial_specs(_trial_specs(), 2)

    assert truncated.inputs["epsilon"].shape == (2, 4, 8)
    assert truncated.inputs["graph_channel"].shape == (2, 4, 2)
    assert truncated.inputs["effector_target"].pos.shape == (2, 4, 2)
    assert truncated.inits["mechanics.vector"].shape == (2, 48)


def test_summarize_loss_tree_and_delta_use_weighted_feedbax_terms() -> None:
    active_tree = TermTree.branch(
        "loss",
        {
            "a": TermTree.leaf("a", jnp.array([1.0, 3.0]), weight=2.0),
            "b": TermTree.leaf("b", jnp.array([5.0]), weight=0.5),
        },
    )
    zero_tree = TermTree.branch(
        "loss",
        {
            "a": TermTree.leaf("a", jnp.array([1.0, 1.0]), weight=2.0),
            "b": TermTree.leaf("b", jnp.array([1.0]), weight=0.5),
        },
    )

    active = summarize_loss_tree(active_tree)
    zero = summarize_loss_tree(zero_tree)
    delta = loss_delta_summary(active, zero)

    assert active["terms"]["a"] == 4.0
    assert active["terms"]["b"] == 2.5
    assert active["total"] == 6.5
    assert delta["terms"]["a"] == 2.0
    assert delta["terms"]["b"] == 2.0
    assert delta["total"] == 4.0


def test_gradient_pair_metrics_report_norms_and_cosine() -> None:
    active = {"w": jnp.array([3.0, 4.0]), "b": jnp.array([1.0])}
    zero = {"w": jnp.array([0.0, 4.0]), "b": jnp.array([1.0])}

    metrics = gradient_pair_metrics(active, zero)

    assert metrics["active_gradient_norm"] == np.sqrt(26.0)
    assert metrics["zero_gradient_norm"] == np.sqrt(17.0)
    assert metrics["active_minus_zero_gradient_norm"] == 3.0
    assert 0.0 < metrics["active_zero_gradient_cosine"] < 1.0


def test_epsilon_summary_and_csv_schema_are_unambiguous() -> None:
    epsilon = np.ones((2, 3, 4), dtype=np.float64)
    summary = epsilon_summary(epsilon)

    assert summary["shape"] == [2, 3, 4]
    assert summary["per_trial_l2"]["mean"] == np.sqrt(12.0)
    assert summary["all_zero"] is False

    csv_text = render_summary_csv(
        [
            {
                "run_id": "run",
                "n_rollout_trials": 2,
                "broad_epsilon_training": {"level": "strong"},
                "epsilon": {
                    "active_total": summary,
                    "paired_without_broad": summary,
                    "broad_delta": summary,
                },
                "loss": {
                    "active": {"total": 3.0},
                    "zero": {"total": 2.0},
                    "delta_active_minus_zero": {"total": 1.0},
                },
                "gradient": {
                    "status": "evaluated",
                    "aggregate": {"relative_delta_norm_vs_active": {"mean": 0.25}},
                },
            }
        ]
    )
    rows = list(csv.DictReader(StringIO(csv_text)))

    assert rows[0]["run_id"] == "run"
    assert rows[0]["gradient_status"] == "evaluated"
