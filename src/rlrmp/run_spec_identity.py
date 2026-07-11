"""Stable identity helpers for generated RLRMP run-spec payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from feedbax.contracts.manifest import canonical_json_bytes, sha256_bytes


RUN_SPEC_IDENTITY_KEYS = (
    "adversarial_phase",
    "artifact_output_dir",
    "batch_size",
    "checkpointing",
    "consumed_data_identities",
    "controller_lr",
    "delayed_reach",
    "fidelity_status",
    "full_training_launch",
    "hps",
    "issue",
    "loss_objective",
    "modal_launch",
    "mode",
    "model_summary",
    "n_train_batches",
    "nominal_only",
    "optimizer",
    "rlrmp_run_spec",
    "schema_version",
    "seed",
    "spec_dir",
    "stochastic_preset",
    "training_diagnostics",
    "training_distribution",
    "training_script",
    "training_summary",
    "validation_bins",
)

STOCHASTIC_FLOAT_PRECISION = {
    "diag_first_block": 6,
    "initial_diag_first_block": 6,
    "noise_std": 7,
    "sensory_noise_std": 7,
    "sensory_covariance_diag": 7,
}


def normalize_run_spec_payload(value: Any, *, key: str | None = None) -> Any:
    """Normalize environment-specific and stochastic leaves before hashing."""
    if key == "rlrmp_commit":
        return "<current-commit>"
    if key == "rlrmp_branch":
        return "<current-branch>"
    if isinstance(value, Mapping):
        return {
            child_key: normalize_run_spec_payload(item, key=child_key)
            for child_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_run_spec_payload(item, key=key) for item in value]
    if isinstance(value, float) and key in STOCHASTIC_FLOAT_PRECISION:
        return float(f"{value:.{STOCHASTIC_FLOAT_PRECISION[key]}g}")
    if isinstance(value, str) and Path(value).is_absolute():
        return _normalize_absolute_path(value)
    return value


def stable_run_spec_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Select and normalize the stable generated run-spec contract."""
    return normalize_run_spec_payload({key: payload[key] for key in RUN_SPEC_IDENTITY_KEYS})


def run_spec_payload_identity_sha256(payload: Mapping[str, Any]) -> str:
    """Return the strict canonical-JSON SHA-256 of a generated run spec."""
    normalized = stable_run_spec_payload(payload)
    json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_bytes(canonical_json_bytes(normalized))


def run_spec_semantic_checks(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract a compact, readable set of load-bearing run-spec semantics."""
    hps = payload["hps"]
    return {
        "seed": payload["seed"],
        "loss_objective": payload["loss_objective"],
        "mode": payload["mode"],
        "schema_version": payload["schema_version"],
        "adversarial_phase": payload["adversarial_phase"],
        "hps": {
            "batch_size": hps["batch_size"],
            "n_batches_baseline": hps["n_batches_baseline"],
            "n_batches_condition": hps["n_batches_condition"],
            "learning_rate_0": hps["learning_rate_0"],
            "model": {
                "hidden_size": hps["model"]["hidden_size"],
                "n_replicates": hps["model"]["n_replicates"],
            },
            "delayed_reach": {"enabled": hps["delayed_reach"]["enabled"]},
            "broad_epsilon_pgd_training": {
                "enabled": hps["broad_epsilon_pgd_training"]["enabled"],
                "mode": hps["broad_epsilon_pgd_training"]["mode"],
            },
            "target_relative_multitarget": {
                "enabled": hps["target_relative_multitarget"]["enabled"],
                "mode": hps["target_relative_multitarget"]["mode"],
            },
        },
    }


def _normalize_absolute_path(value: str) -> str:
    marker = "/worktrees/"
    if marker in value:
        prefix, suffix = value.split(marker, 1)
        worktree_parts = suffix.split("/", 1)
        if len(worktree_parts) == 2 and prefix.endswith("/rlrmp"):
            return f"<repo-root>/{worktree_parts[1]}"
    repo_marker = "/rlrmp/"
    if repo_marker in value:
        return f"<repo-root>/{value.rsplit(repo_marker, 1)[1]}"
    return "<absolute-path>"
