from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from feedbax.artifact_schema import read_npz_array_store, write_npz_array_store

from rlrmp.artifact_migration import (
    discover_b_set_runs,
    extract_role_addressed_arrays,
    minimax_args_from_run_spec,
    validate_array_store_roundtrip,
)


class _Nested(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray


class _TinyModel(eqx.Module):
    nodes: dict[str, _Nested]


def test_extract_role_addressed_arrays_uses_stable_tree_paths(tmp_path: Path) -> None:
    model = _TinyModel(
        nodes={
            "net": _Nested(
                weight=jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
                bias=jnp.ones((2,), dtype=jnp.float32),
            )
        }
    )

    arrays = extract_role_addressed_arrays(model, root_role="model")
    assert sorted(arrays) == [
        "model.nodes.net.bias",
        "model.nodes.net.weight",
    ]

    store_path = tmp_path / "tiny.arrays.npz"
    write_npz_array_store(store_path, arrays, store_role="params")
    validation = validate_array_store_roundtrip(store_path, arrays)
    loaded = read_npz_array_store(store_path)

    assert validation.status == "passed"
    assert loaded.payload.roles == ["model.nodes.net.bias", "model.nodes.net.weight"]


def test_minimax_args_from_run_spec_normalizes_historical_cli_flags() -> None:
    args = minimax_args_from_run_spec(
        {
            "cli_flags": {
                "--hidden-type": "gru",
                "--n-warmup-batches": 12000,
                "--no-streaming-loss": True,
                "nn_hidden_derivative": 0.001,
            },
            "controller_lr": 1e-4,
        }
    )

    assert args.hidden_type == "gru"
    assert args.n_warmup_batches == 12000
    assert args.streaming_loss is False
    assert args.nn_hidden_derivative == 0.001
    assert args.n_adversary_batches == 0


def test_discover_b_set_runs_handles_flat_and_nested_layouts(tmp_path: Path) -> None:
    flat_spec = tmp_path / "results" / "2bc95fd" / "runs" / "gru__jerk.json"
    nested_spec = tmp_path / "results" / "f47abb1" / "runs" / "lit__full_nojerk" / "run.json"
    flat_artifact = tmp_path / "_artifacts" / "2bc95fd" / "gru__jerk"
    nested_artifact = tmp_path / "_artifacts" / "f47abb1" / "runs" / "lit__full_nojerk"
    for path in [flat_spec, nested_spec]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cli_flags": {"--hidden-type": "gru"}}), encoding="utf-8")
    for path in [flat_artifact, nested_artifact]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "warmup_model.eqx").write_bytes(b"placeholder")

    runs = discover_b_set_runs(tmp_path)
    by_id = {(run.issue_id, run.run_label): run for run in runs}

    assert by_id[("2bc95fd", "gru__jerk")].artifact_dir == flat_artifact
    assert by_id[("f47abb1", "lit__full_nojerk")].artifact_dir == nested_artifact
