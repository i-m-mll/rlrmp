"""One-time migration for 9727d79 legacy guided-distillation model files.

The completed run originally serialized a standalone policy that consumed
``[feedback, previous_action]``. Standard C&S h0 GRU materializers expect the
normal Feedbax graph from ``setup_task_model_pair`` and a controller input of
the 6D force-filter feedback only. This helper exists only to recover the
completed local artifact into the standard-loadable graph form; production
training and evaluation paths must not import this legacy template.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax_cookbook import load_with_hyperparameters

from rlrmp.train.distillation_native.guided_kernel import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEC_PATH,
    _save_pytree,
    _standard_hps_from_spec,
    standard_controller_action_dim,
    standard_controller_feedback_dim,
    standard_controller_parts,
)


class LegacyGuidedDistillationPolicy(eqx.Module):
    """Legacy standalone student policy used only for artifact migration."""

    h0_encoder: eqx.nn.Linear
    cell: eqx.nn.GRUCell
    readout: eqx.nn.Linear

    def __init__(
        self,
        *,
        feedback_dim: int,
        action_dim: int,
        hidden_size: int,
        key: jax.Array,
    ) -> None:
        h0_key, cell_key, readout_key = jr.split(key, 3)
        self.h0_encoder = eqx.nn.Linear(
            feedback_dim,
            hidden_size,
            dtype=jnp.float32,
            key=h0_key,
        )
        self.cell = eqx.nn.GRUCell(
            feedback_dim + action_dim,
            hidden_size,
            dtype=jnp.float32,
            key=cell_key,
        )
        self.readout = eqx.nn.Linear(
            hidden_size,
            action_dim,
            dtype=jnp.float32,
            key=readout_key,
        )


def _setup_task_model_pair(hps: Any, *, key: Any) -> Any:
    import rlrmp.analysis  # noqa: F401
    from rlrmp.train.task_model import setup_task_model_pair

    return setup_task_model_pair(hps, key=key)


def _linear_weight(layer: Any) -> jax.Array:
    """Return the underlying linear weight for plain or masked readout layers."""

    linear = getattr(layer, "linear", layer)
    return linear.weight


def _linear_bias(layer: Any) -> jax.Array:
    """Return the underlying linear bias for plain or masked readout layers."""

    linear = getattr(layer, "linear", layer)
    return linear.bias


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", default=str(DEFAULT_SPEC_PATH))
    parser.add_argument("--artifact-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--migrate-checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    run_spec_path = Path(args.run_spec)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir_resolved = artifact_dir.resolve()
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    hps = _standard_hps_from_spec(
        run_spec,
        n_replicates=int(run_spec["model_contract"]["n_replicates"]),
        hidden_size=int(run_spec["model_contract"]["hidden_size"]),
        batch_size=int(run_spec["model_contract"]["batch_size"]),
        n_batches=int(run_spec["training_schedule"]["total_batches"]),
        controller_lr=float(run_spec["optimizer"]["controller_lr"]),
        lr_warmup_batches=int(run_spec["optimizer"]["lr_warmup_batches"]),
        lr_warmup_init_fraction=float(run_spec["optimizer"]["lr_warmup_init_fraction"]),
        lr_cosine_alpha=float(run_spec["optimizer"]["lr_cosine_alpha"]),
        gradient_clip_norm=float(run_spec["optimizer"]["gradient_clip_norm"]),
    )
    standard_template = _setup_task_model_pair(
        hps,
        key=jr.PRNGKey(int(run_spec.get("seed", 0))),
    ).model
    n_replicates = int(hps.model.n_replicates)
    hidden_size = int(hps.model.hidden_size)
    feedback_dim = standard_controller_feedback_dim(standard_template)
    action_dim = standard_controller_action_dim(standard_template)
    legacy_template = eqx.filter_vmap(
        lambda key: LegacyGuidedDistillationPolicy(
            feedback_dim=feedback_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            key=key,
        )
    )(jr.split(jr.PRNGKey(int(run_spec.get("seed", 0))), n_replicates))

    latest_checkpoint = (artifact_dir / "checkpoints" / "checkpoint_latest").resolve()
    latest_legacy = latest_checkpoint / "models.eqx"
    if not latest_legacy.is_file():
        raise FileNotFoundError(f"Legacy latest checkpoint model not found: {latest_legacy}")
    latest_standard = migrate_one_model(
        legacy_path=latest_legacy,
        legacy_template=legacy_template,
        standard_template=standard_template,
        feedback_dim=feedback_dim,
    )
    _save_pytree(artifact_dir / "trained_model.eqx", latest_standard, hyperparameters=run_spec)

    migrated_checkpoints: list[str] = []
    if args.migrate_checkpoints:
        checkpoint_root = artifact_dir / "checkpoints"
        for checkpoint_dir in sorted(checkpoint_root.glob("checkpoint_[0-9]*")):
            legacy_path = checkpoint_dir / "models.eqx"
            if not legacy_path.is_file():
                continue
            standard_model = migrate_one_model(
                legacy_path=legacy_path,
                legacy_template=legacy_template,
                standard_template=standard_template,
                feedback_dim=feedback_dim,
            )
            eqx.tree_serialise_leaves(checkpoint_dir / "model.eqx", standard_model)
            update_checkpoint_metadata(checkpoint_dir / "metadata.json")
            migrated_checkpoints.append(str(checkpoint_dir.relative_to(artifact_dir)))

    update_training_summary(artifact_dir / "training_summary.json")
    manifest = {
        "schema_version": "rlrmp.9727d79.standard_model_migration.v1",
        "run_spec": str(run_spec_path),
        "artifact_dir": str(artifact_dir),
        "standard_model": "trained_model.eqx",
        "checkpoint_model": "checkpoints/<checkpoint>/model.eqx",
        "legacy_model_sources": {
            "final_checkpoint": str(latest_legacy.relative_to(artifact_dir_resolved)),
            "legacy_final_replicates": [
                f"student_model_rep{replicate}.eqx" for replicate in range(n_replicates)
            ],
        },
        "projection": projection_manifest(feedback_dim=feedback_dim, action_dim=action_dim),
        "migrated_checkpoints": migrated_checkpoints,
        "standard_loader_smoke": smoke_standard_load(
            artifact_dir / "trained_model.eqx",
            hps=hps,
            seed=int(run_spec.get("seed", 0)),
        ),
    }
    manifest_path = artifact_dir / "standard_model_migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest["standard_loader_smoke"]}))
    return 0


def migrate_one_model(
    *,
    legacy_path: Path,
    legacy_template: Any,
    standard_template: Any,
    feedback_dim: int,
) -> Any:
    legacy_model = eqx.tree_deserialise_leaves(legacy_path, legacy_template)
    standard = standard_template
    return eqx.tree_at(
        lambda model: (
            standard_controller_parts(model).h0_encoder.weight,
            standard_controller_parts(model).h0_encoder.bias,
            standard_controller_parts(model).hidden_cell.weight_ih,
            standard_controller_parts(model).hidden_cell.weight_hh,
            standard_controller_parts(model).hidden_cell.bias,
            standard_controller_parts(model).hidden_cell.bias_n,
            _linear_weight(standard_controller_parts(model).readout),
            _linear_bias(standard_controller_parts(model).readout),
        ),
        standard,
        replace=(
            _like(
                legacy_model.h0_encoder.weight,
                standard_controller_parts(standard).h0_encoder.weight,
            ),
            _like(
                legacy_model.h0_encoder.bias,
                standard_controller_parts(standard).h0_encoder.bias,
            ),
            _like(
                legacy_model.cell.weight_ih[..., :feedback_dim],
                standard_controller_parts(standard).hidden_cell.weight_ih,
            ),
            _like(
                legacy_model.cell.weight_hh,
                standard_controller_parts(standard).hidden_cell.weight_hh,
            ),
            _like(legacy_model.cell.bias, standard_controller_parts(standard).hidden_cell.bias),
            _like(
                legacy_model.cell.bias_n,
                standard_controller_parts(standard).hidden_cell.bias_n,
            ),
            _like(
                legacy_model.readout.weight,
                _linear_weight(standard_controller_parts(standard).readout),
            ),
            _like(
                legacy_model.readout.bias,
                _linear_bias(standard_controller_parts(standard).readout),
            ),
        ),
    )


def _like(value: Any, template: Any) -> Any:
    return jnp.asarray(value, dtype=template.dtype)


def update_checkpoint_metadata(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["model_path"] = "model.eqx"
    metadata["legacy_model_path"] = "models.eqx"
    metadata["standard_model_migration"] = projection_manifest(
        feedback_dim=6,
        action_dim=2,
    )
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def update_training_summary(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    artifacts = dict(summary.get("artifacts") or {})
    legacy_replicates = artifacts.pop(
        "student_model_replicates",
        artifacts.get("legacy_student_model_replicates", []),
    )
    if not legacy_replicates:
        legacy_replicates = sorted(
            child.name for child in path.parent.glob("student_model_rep*.eqx")
        )
    artifacts["trained_model"] = "trained_model.eqx"
    artifacts["legacy_student_model_replicates"] = legacy_replicates
    artifacts["standard_model_migration_manifest"] = "standard_model_migration_manifest.json"
    summary["artifacts"] = artifacts
    summary["model_contract"] = "standard_feedbax_graph"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def projection_manifest(*, feedback_dim: int, action_dim: int) -> dict[str, Any]:
    return {
        "status": "standard_loadable_lossy_projection",
        "reason": (
            "Legacy standalone policy consumed feedback concatenated with previous action; "
            "the standard Feedbax h0 graph consumes controller feedback only."
        ),
        "copied_leaves": [
            "nodes.net.h0_encoder",
            "nodes.net.net.hidden.weight_hh",
            "nodes.net.net.hidden.bias",
            "nodes.net.net.hidden.bias_n",
            "nodes.net.net.readout.linear",
        ],
        "projected_leaf": "nodes.net.net.hidden.weight_ih",
        "kept_input_columns": list(range(feedback_dim)),
        "dropped_legacy_action_history_columns": list(
            range(feedback_dim, feedback_dim + action_dim)
        ),
    }


def smoke_standard_load(model_path: Path, *, hps: Any, seed: int) -> dict[str, Any]:
    model, _hyperparameters = load_with_hyperparameters(
        model_path,
        setup_func=lambda key, **_kwargs: _setup_task_model_pair(hps, key=key).model,
    )
    return {
        "load_status": "ok",
        "model_path": str(model_path),
        "n_replicates": int(standard_controller_parts(model).hidden_cell.weight_ih.shape[0]),
        "feedback_dim": standard_controller_feedback_dim(model),
        "seed": int(seed),
    }


if __name__ == "__main__":
    raise SystemExit(main())
