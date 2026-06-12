"""Load RLRMP models migrated to Feedbax model-artifact manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.random as jr
from feedbax.artifact_materialize import materialize_model_artifact
from feedbax.manifest import ModelArtifactManifest, ParentRef

from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax import build_hps


LEGACY_EXECUTION_BACKEND = "rlrmp.legacy_simple_feedback_compat"


_DEFAULT_MINIMAX_ARGS: dict[str, Any] = {
    "n_warmup_batches": 12000,
    "batch_size": 250,
    "n_replicates": 5,
    "n_adversary_batches": 0,
    "n_adversary_steps": 5,
    "adversary_lr": 3e-4,
    "controller_lr": 1e-4,
    "adversary_type": "linear_dynamics",
    "linear_dynamics_eta_max": 0.1,
    "linear_dynamics_pgd_steps": 5,
    "linear_dynamics_lr": 1e-2,
    "n_bumps": 3,
    "force_max": 1.0,
    "n_adversaries": 1,
    "adv_batch_size": None,
    "warmup_model": None,
    "output_dir": None,
    "spec_dir": None,
    "jax_cache_dir": None,
    "jax_explain_cache_misses": False,
    "seed": 42,
    "checkpoint": True,
    "checkpoint_every": 1000,
    "resume": False,
    "loss_update_enabled": False,
    "loss_update_ratio": 0.5,
    "fused": True,
    "streaming_loss": False,
    "hidden_type": "gru",
    "nn_hidden_derivative": 0.0,
    "nn_output_jerk": 0.0,
    "nn_output_pre_go": 0.0,
    "nn_hidden_derivative_pre_go": 0.0,
    "sisu_gating": "additive",
    "effector_hold_pos": 10.0,
    "effector_hold_vel": 10.0,
    "effector_final_vel": 0.0,
    "effector_vel_late": 0.1,
    "effector_pos_running": 1.0,
    "effector_pos_late_weight": 0.5,
    "effector_pos_late_final_scale": 2.0,
    "effector_pos_late_start_step": 80,
    "effector_pos_running_schedule": "flat",
    "effector_hold_pos_schedule": "flat",
    "position_powerlaw_power": 6.0,
    "movement_ramp_shape": "linear",
    "movement_ramp_duration_steps": 60,
    "movement_ramp_power": 2.0,
    "p_catch_trial": 0.5,
    "nn_output": 1e-5,
    "nn_hidden": 1e-5,
}


def minimax_args_from_run_spec(run_spec: dict[str, Any]) -> argparse.Namespace:
    """Build a minimax argparse namespace from historical run-spec JSON."""

    values = dict(_DEFAULT_MINIMAX_ARGS)
    for key, value in _normalized_cli_flags(run_spec).items():
        if key in values:
            values[key] = value
    for key, value in run_spec.items():
        if key in values and key not in {"cli_flags", "feedbax_graph"}:
            values[key] = value
    return argparse.Namespace(**values)


def load_migrated_model_artifact(
    manifest_path: Path | str,
    *,
    repo_root: Path | str = Path("."),
    key=None,
) -> Any:
    """Load an RLRMP migrated model artifact from its Feedbax manifest.

    The artifacts produced for ``b41c940`` declare the execution backend as
    ``rlrmp.legacy_simple_feedback_compat``. Rehydration therefore uses the
    legacy run spec recorded in the manifest to rebuild the executable template,
    then fills that template from the Feedbax role-addressed array store.
    """

    root = Path(repo_root)
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    manifest = ModelArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))

    backend = (manifest.graph_spec.metadata or {}).get("execution_backend")
    if backend != LEGACY_EXECUTION_BACKEND:
        raise ValueError(
            f"Unsupported migrated artifact execution backend {backend!r}; "
            f"expected {LEGACY_EXECUTION_BACKEND!r}."
        )

    run_spec_parent = _manifest_parent(manifest, "legacy_run_spec")
    run_spec_path = root / run_spec_parent.uri
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    hps = build_hps(minimax_args_from_run_spec(run_spec))

    template_key = jr.PRNGKey(0) if key is None else key
    template = setup_task_model_pair(hps, key=template_key).model

    if manifest.parameter_store is None:
        raise ValueError(f"Manifest {path} has no parameter_store.")
    return materialize_model_artifact(path, template, root=root, root_role="model")


def _manifest_parent(manifest: ModelArtifactManifest, kind: str) -> ParentRef:
    matches = [parent for parent in manifest.provenance.parents if parent.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {kind!r} parent in manifest {manifest.id}; found {len(matches)}."
        )
    return matches[0]


def _normalized_cli_flags(run_spec: dict[str, Any]) -> dict[str, Any]:
    flags = run_spec.get("cli_flags") or {}
    normalized: dict[str, Any] = {}
    for key, value in flags.items():
        normalized_key = str(key).lstrip("-").replace("-", "_")
        if normalized_key.startswith("no_") and value is True:
            normalized[normalized_key[3:]] = False
        else:
            normalized[normalized_key] = value
    return normalized
