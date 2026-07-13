"""Fixed pre-retirement trajectory oracles captured from commit 0ee30e6f."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import pytest
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.train.distillation_entry import load_distillation_run_spec
from rlrmp.train.distillation_native import execute_distillation_training_run_spec_native
from rlrmp.train.minimax_native import (
    MinimaxControllerState,
    build_hps,
    build_minimax_training_run_spec,
    execute_minimax_training_run_spec_native,
)
from rlrmp.train.task_model import build_task_base
from rlrmp.train.training_configs import (
    ClosedLoopDistillationConfig,
    GuidedDistillationConfig,
    MinimaxConfig,
)

ORACLE_ROOT = Path("tests/fixtures/pre_native_oracles/v1")
ORACLE_MANIFEST = json.loads((ORACLE_ROOT / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _oracle_uses_captured_x64_mode():
    """Match the float32 JAX mode used to capture the frozen 0ee30e6f oracles."""

    with jax.enable_x64(False):
        yield


def _minimax_spec(tmp_path: Path, adversary_type: str) -> TrainingRunSpec:
    config = MinimaxConfig(
        adversary_type=adversary_type,
        n_warmup_batches=0,
        n_adversary_batches=2,
        n_adversary_steps=1,
        batch_size=1,
        adv_batch_size=1,
        n_replicates=1,
        output_dir=str(tmp_path / adversary_type),
    )
    hps = build_hps(config)
    graph_bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
        key=jr.PRNGKey(config.seed),
    )
    payload = build_minimax_training_run_spec(
        config,
        graph_spec=graph_bundle.graph_spec,
        output_dir=Path(config.output_dir),
        spec_dir=tmp_path / "spec",
        feedbax_graph={"graph_spec_path": "graph_spec.json", "manifest_path": "manifest.json"},
    )
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _minimax_comparable(slots: dict[str, Any]) -> dict[str, Any]:
    controller = slots["controller"]
    assert isinstance(controller, MinimaxControllerState)
    comparable = {
        "controller": controller.per_replicate_leaves,
        "controller_optimizer": slots["controller_optimizer"],
        "adversary_population": slots["adversary_population"],
        "adversary_optimizer": slots["adversary_optimizer"],
        "rng": slots["rng"],
        "controller_loss": slots["controller_loss"],
        "adversary_loss": slots["adversary_loss"],
    }
    return jt.map(
        lambda leaf: (
            leaf.astype(jnp.int8)
            if getattr(getattr(leaf, "dtype", None), "kind", None) == "b"
            else leaf
        ),
        comparable,
    )


def _candidate_arrays(tree: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path, leaf in jt.flatten_with_path(tree)[0]:
        try:
            array = np.asarray(jax.device_get(leaf))
        except Exception:
            continue
        if array.dtype.kind in "biufc":
            arrays[jax.tree_util.keystr(path)] = array
    return arrays


def _assert_matches_oracle(case_name: str, tree: Any) -> None:
    case = ORACLE_MANIFEST["cases"][case_name]
    expected = np.load(ORACLE_ROOT / case["file"])
    candidate = _candidate_arrays(tree)
    expected_paths = {leaf["path"] for leaf in case["leaves"]}
    assert set(candidate) == expected_paths
    for leaf in case["leaves"]:
        actual = candidate[leaf["path"]]
        target = expected[leaf["key"]]
        assert list(actual.shape) == leaf["shape"]
        if actual.dtype.kind in "fc":
            np.testing.assert_allclose(
                actual,
                target,
                atol=case["tolerance"]["atol"],
                rtol=case["tolerance"]["rtol"],
            )
        else:
            np.testing.assert_array_equal(actual, target)


@pytest.mark.parametrize(
    ("adversary_type", "seed"),
    [("gaussian_bump", 1), ("linear_dynamics", 3)],
)
def test_minimax_executor_matches_pre_native_oracle(
    tmp_path: Path,
    adversary_type: str,
    seed: int,
) -> None:
    spec = _minimax_spec(tmp_path, adversary_type)
    full = execute_minimax_training_run_spec_native(
        spec,
        run_id=f"oracle-minimax-{adversary_type}",
        key=jr.PRNGKey(seed),
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    stopped = execute_minimax_training_run_spec_native(
        spec,
        run_id=f"oracle-minimax-{adversary_type}-resume",
        key=jr.PRNGKey(seed),
        manifest_root=tmp_path / "manifests" / "stopped",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_adversarial",
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_minimax_training_run_spec_native(
        spec,
        run_id=f"oracle-minimax-{adversary_type}-resume",
        key=jr.PRNGKey(seed),
        manifest_root=tmp_path / "manifests" / "resumed",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )
    assert stopped.final_coordinate.completed_barrier == "after_adversarial"
    _assert_matches_oracle(
        f"minimax_{adversary_type}",
        {
            "full": _minimax_comparable(full.final_slots),
            "resumed": _minimax_comparable(resumed.final_slots),
        },
    )


@pytest.mark.parametrize(
    ("method", "barrier"),
    [
        ("guided_distillation", "after_teacher_forced_warm_start"),
        ("closed_loop_distillation", "after_closed_loop_rollout_distillation"),
    ],
)
def test_distillation_executor_matches_pre_native_oracle(
    tmp_path: Path,
    method: str,
    barrier: str,
) -> None:
    if method == "guided_distillation":
        config = GuidedDistillationConfig(
            run_spec="tests/fixtures/legacy_payloads/guided_distillation_run_spec.json",
            n_batches=1,
            batch_size=1,
            n_replicates=1,
            hidden_size=6,
            n_jvp_directions=1,
            output_dir=str(tmp_path / "guided"),
            checkpoint=False,
        )
    else:
        config = ClosedLoopDistillationConfig(
            run_spec=Path(
                "tests/fixtures/legacy_payloads/closed_loop_distillation_run_spec.json"
            ),
            n_batches=1,
            batch_size=1,
            n_replicates=1,
            hidden_size=6,
            output_dir=str(tmp_path / "closed"),
        )
    spec = load_distillation_run_spec(config, method=method)
    full = execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=f"oracle-{method}",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    stopped = execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=f"oracle-{method}-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "stopped",
        checkpoint_root=checkpoint_root,
        stop_after_barrier=barrier,
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=f"oracle-{method}-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "resumed",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )
    assert stopped.final_coordinate.completed_barrier == barrier
    keys = ("model", "optimizer", "prng", "completed_batches", "train_loss")
    _assert_matches_oracle(
        method,
        {
            "full": {key: full.final_slots[key] for key in keys if key in full.final_slots},
            "resumed": {
                key: resumed.final_slots[key] for key in keys if key in resumed.final_slots
            },
        },
    )
