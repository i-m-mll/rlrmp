"""Materialize the 9bb676f regenerated frozen-policy Adam/lambda gate."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
from feedbax.runtime.batch import BatchInfo

from rlrmp.analysis.frozen_adversary_audit import (
    AFFINE_POLICY,
    LINEAR_NO_BIAS_POLICY,
    finite_policy_epsilon_from_parameters,
    realized_epsilon_energy,
)
from rlrmp.analysis.frozen_policy_gate import (
    DIRECT_EPSILON_MECHANISM,
    FrozenAuditRow,
    FrozenBatchDescriptor,
    FrozenOptimizerConfig,
    directional_curvature_summary,
    metric_geometry_summary,
    selected_epsilon_invariance,
    sha256_file,
    sha256_json,
)
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT
from rlrmp.train.closed_loop_finite_adversary import target_centered_full_state_features
from rlrmp.train.cs_nominal_gru import (
    _args_values_from_run_spec,
    _is_replicate_axis_array,
    _with_single_replicate_state_initializers,
    build_hps,
    build_parser,
)
from rlrmp.train.cs_perturbation_training import (
    _broad_epsilon_pgd_trust_radius,
    _ensure_broad_epsilon_input,
    _epsilon_time_mask,
    _flattened_per_trial_norm,
    _project_flattened_per_trial_l2_ball,
    _set_input,
    _trial_target_position_m,
    config_from_broad_epsilon_pgd_hps,
)
from rlrmp.train.task_model import setup_task_model_pair

ISSUE = "9bb676f"
SOURCE_ISSUE = "ae9f30f"
SOURCE_RUN = "linear_no_bias_b1p05"
CHECKPOINT_BATCHES = (500, 12000)
MECHANISMS = (DIRECT_EPSILON_MECHANISM, LINEAR_NO_BIAS_POLICY, AFFINE_POLICY)
AUDIT_BATCH_SIZE = 4
REPLICATE_INDEX = 0
FROZEN_ADAM_STEPS = 12
FROZEN_ADAM_LR = 1e-3
ROOT_KEY_SEED = 9_676


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=AUDIT_BATCH_SIZE)
    parser.add_argument("--adam-steps", type=int, default=FROZEN_ADAM_STEPS)
    parser.add_argument("--adam-lr", type=float, default=FROZEN_ADAM_LR)
    parser.add_argument(
        "--checkpoint-batches",
        type=int,
        nargs="+",
        default=list(CHECKPOINT_BATCHES),
    )
    args = parser.parse_args()

    results_dir = REPO_ROOT / "results" / ISSUE
    notes_dir = results_dir / "notes"
    artifact_dir = REPO_ROOT / "_artifacts" / ISSUE / "frozen_policy_gate"
    results_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    lambda_source_path = REPO_ROOT / "results" / "06a4dc8" / "canonical_soft_lambda_hvp.json"
    lambda_source = json.loads(lambda_source_path.read_text(encoding="utf-8"))
    lambda_input = _candidate_lambda(lambda_source, beta=1.05)
    source_root = REPO_ROOT / "_artifacts" / SOURCE_ISSUE / "runs" / SOURCE_RUN / "checkpoints"

    descriptors: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for checkpoint_batches in args.checkpoint_batches:
        checkpoint = source_root / f"checkpoint_{int(checkpoint_batches):07d}"
        result = _audit_checkpoint(
            checkpoint,
            lambda_input=lambda_input,
            batch_size=int(args.batch_size),
            adam_steps=int(args.adam_steps),
            adam_lr=float(args.adam_lr),
            artifact_dir=artifact_dir,
        )
        descriptors.append(result["descriptor"])
        rows.extend(result["rows"])

    payload = {
        "schema_version": "rlrmp.frozen_policy_adam_lambda_gate.v1",
        "issue": ISSUE,
        "source_issue": SOURCE_ISSUE,
        "source_run": SOURCE_RUN,
        "checkpoint_policy": {
            "selected": list(args.checkpoint_batches),
            "reason": (
                "500 batches captures the prior early nonzero finite-policy behavior; "
                "12000 batches captures the completed late suppression/final checkpoint."
            ),
        },
        "lambda_source": {
            "path": _repo_relative(lambda_source_path),
            "sha256": sha256_file(lambda_source_path),
            "beta": 1.05,
            "lambda": lambda_input,
            "mapping": "beta^2 * pooled p90 lambda_star from 06a4dc8",
        },
        "optimizer": FrozenOptimizerConfig(
            method="adam",
            learning_rate=float(args.adam_lr),
            n_steps=int(args.adam_steps),
        ).to_json(),
        "descriptors": descriptors,
        "rows": rows,
        "mechanism_lambda_recommendations": _lambda_recommendations(rows, lambda_input),
        "readiness": _readiness(rows),
    }

    json_path = results_dir / "frozen_policy_adam_lambda_gate.json"
    write_compact_json(json_path, payload)
    _write_readme(results_dir)
    _write_markdown(notes_dir / "frozen_policy_adam_lambda_gate.md", payload)
    print(json.dumps({"json": _repo_relative(json_path), "rows": len(rows)}, indent=2))


def _audit_checkpoint(
    checkpoint: Path,
    *,
    lambda_input: float,
    batch_size: int,
    adam_steps: int,
    adam_lr: float,
    artifact_dir: Path,
) -> dict[str, Any]:
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_spec = metadata["run_spec"]
    args = build_parser().parse_args([])
    for key, value in _args_values_from_run_spec(run_spec).items():
        setattr(args, key, value)
    args.batch_size = int(batch_size)

    hps = build_hps(args)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", 42))))
    model = eqx.tree_deserialise_leaves(checkpoint / "model.eqx", pair.model)
    model = _replicate_model(model, hps, REPLICATE_INDEX)
    broad_cfg = config_from_broad_epsilon_pgd_hps(hps.broad_epsilon_pgd_training)

    root_key = jr.fold_in(jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32), ROOT_KEY_SEED)
    root_key = jr.fold_in(root_key, int(metadata["completed_batches"]))
    key_trials, key_model, key_optimizer = jr.split(root_key, 3)
    batch_index = max(0, int(metadata["completed_batches"]) - 1)
    batch_info = BatchInfo(
        size=int(batch_size),
        start=jnp.asarray(0),
        current=jnp.asarray(batch_index),
        total=jnp.asarray(hps.n_batches_condition),
    )
    keys_trials = jr.split(key_trials, int(batch_size))
    keys_model = jr.split(key_model, int(batch_size))
    trial_specs = eqx.filter_vmap(
        partial(pair.task.get_train_trial_with_intervenor_params, batch_info=batch_info)
    )(keys_trials)
    specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=int(broad_cfg.epsilon_dim))
    base_epsilon = jnp.asarray(specs.inputs["epsilon"])
    clean_states = pair.task.eval_trials(model, specs, keys_model)
    mechanics = jnp.asarray(clean_states.mechanics.vector)[..., : int(hps.model.state_dim)]
    target = _trial_target_position_m(specs)
    features = target_centered_full_state_features(mechanics, target_position=target)
    radius = _broad_epsilon_pgd_trust_radius(specs, broad_cfg).astype(base_epsilon.dtype)
    time_mask = _epsilon_time_mask(specs, base_epsilon, bool(broad_cfg.movement_epoch_only))

    descriptor = FrozenBatchDescriptor(
        source_issue=SOURCE_ISSUE,
        source_run=SOURCE_RUN,
        checkpoint_path=_repo_relative(checkpoint),
        checkpoint_batches=int(metadata["completed_batches"]),
        checkpoint_metadata_sha256=sha256_file(metadata_path),
        run_spec_sha256=sha256_json(run_spec),
        replicate_index=REPLICATE_INDEX,
        batch_size=int(batch_size),
        batch_index=batch_index,
        root_key=_key_list(root_key),
        key_trials=_key_list(key_trials),
        key_model=_key_list(key_model),
        key_optimizer=_key_list(key_optimizer),
        task_distribution={
            "hps_task": run_spec.get("hps", {}).get("task", {}),
            "training_distribution": run_spec.get("training_distribution", {}),
        },
    ).to_json()

    rows = []
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        row_key = jr.fold_in(key_optimizer, mechanism_index)
        rows.append(
            _audit_mechanism(
                mechanism,
                task=pair.task,
                model=model,
                specs=specs,
                keys_model=keys_model,
                base_epsilon=base_epsilon,
                features=features,
                target=target,
                radius=radius,
                time_mask=time_mask,
                lambda_input=lambda_input,
                adam_steps=adam_steps,
                adam_lr=adam_lr,
                row_key=row_key,
                checkpoint_batches=int(metadata["completed_batches"]),
                artifact_dir=artifact_dir,
            )
        )
    return {"descriptor": descriptor, "rows": rows}


def _audit_mechanism(
    mechanism: str,
    *,
    task: Any,
    model: Any,
    specs: Any,
    keys_model: Any,
    base_epsilon: jnp.ndarray,
    features: jnp.ndarray,
    target: jnp.ndarray,
    radius: jnp.ndarray,
    time_mask: jnp.ndarray,
    lambda_input: float,
    adam_steps: int,
    adam_lr: float,
    row_key: jnp.ndarray,
    checkpoint_batches: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    del row_key
    zero_params = _zero_params(mechanism, base_epsilon, features)
    time_mask_vector = np.asarray(time_mask[0, :, 0]) if time_mask.ndim == 3 else None

    def params_to_delta(params):
        if mechanism == DIRECT_EPSILON_MECHANISM:
            raw_delta = jnp.asarray(params)
        elif mechanism == LINEAR_NO_BIAS_POLICY:
            raw_delta = finite_policy_epsilon_from_parameters(
                features,
                params,
                policy_class=LINEAR_NO_BIAS_POLICY,
            )
        elif mechanism == AFFINE_POLICY:
            raw_delta = finite_policy_epsilon_from_parameters(
                features,
                params[..., :-1],
                bias=params[..., -1],
                policy_class=AFFINE_POLICY,
            )
        else:
            raise ValueError(f"unknown mechanism {mechanism!r}")
        return _project_flattened_per_trial_l2_ball(raw_delta * time_mask, radius) * time_mask

    def task_loss_for_params(params):
        delta = params_to_delta(params)
        candidate = _set_input(specs, "epsilon", base_epsilon + delta)
        states = task.eval_trials(model, candidate, keys_model)
        return jnp.asarray(task.loss_func(states, candidate, model).total)

    def energy_for_params(params):
        return realized_epsilon_energy(params_to_delta(params), time_mask=time_mask_vector)

    def objective_for_params(params):
        task_loss = task_loss_for_params(params)
        energy = energy_for_params(params)
        return task_loss - jnp.asarray(lambda_input, dtype=task_loss.dtype) * energy

    zero_task_loss, grad = jax.value_and_grad(task_loss_for_params)(zero_params)
    zero_energy = energy_for_params(zero_params)
    objective_zero = (
        zero_task_loss - jnp.asarray(lambda_input, dtype=zero_task_loss.dtype) * zero_energy
    )
    metric_summary, pressure = metric_geometry_summary(
        mechanism,  # type: ignore[arg-type]
        features=features,
        epsilon_dim=int(base_epsilon.shape[-1]),
        gradient=grad,
        radius=float(np.mean(np.asarray(radius))),
        time_mask=time_mask_vector,
    )
    curvature = _directional_hvp(
        task_loss_for_params,
        energy_for_params,
        zero_params,
        grad,
    )
    selected_params, selected_objective, nonfinite = _run_adam(
        objective_for_params,
        zero_params,
        n_steps=adam_steps,
        learning_rate=adam_lr,
    )
    selected_delta = params_to_delta(selected_params)
    selected_energy = energy_for_params(selected_params)
    tensor_path = (
        artifact_dir / f"checkpoint_{int(checkpoint_batches):07d}" / f"{mechanism}_tensors.npz"
    )
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tensor_path,
        base_epsilon=np.asarray(jax.device_get(base_epsilon)),
        selected_epsilon=np.asarray(jax.device_get(selected_delta)),
        selected_params=np.asarray(jax.device_get(selected_params)),
        zero_gradient=np.asarray(jax.device_get(grad)),
        features=np.asarray(jax.device_get(features)),
        target_position=np.asarray(jax.device_get(target)),
        radius=np.asarray(jax.device_get(radius)),
    )

    row = FrozenAuditRow(
        mechanism=mechanism,  # type: ignore[arg-type]
        lambda_input=float(lambda_input),
        objective_at_zero=float(np.asarray(objective_zero)),
        task_loss_at_zero=float(np.asarray(zero_task_loss)),
        gradient_norm=float(np.linalg.norm(np.asarray(jax.device_get(grad)))),
        gradient_pressure_scale=float(pressure),
        metric_geometry=metric_summary,
        curvature=curvature,
        optimizer=FrozenOptimizerConfig(
            method="adam",
            learning_rate=adam_lr,
            n_steps=adam_steps,
        ).to_json(),
        selected_energy=float(np.asarray(selected_energy)),
        selected_objective=float(np.asarray(selected_objective)),
        accepted_objective_gain=float(np.asarray(selected_objective - objective_zero)),
        cap_behavior=_cap_behavior(selected_delta, radius),
        nonfinite=nonfinite,
        batch_size_invariance=selected_epsilon_invariance(
            selected_delta,
            time_mask=time_mask_vector,
        ),
        tensor_artifact=_repo_relative(tensor_path),
    )
    return row.to_json()


def _zero_params(mechanism: str, base_epsilon: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
    if mechanism == DIRECT_EPSILON_MECHANISM:
        return jnp.zeros_like(base_epsilon)
    horizon = int(base_epsilon.shape[-2])
    epsilon_dim = int(base_epsilon.shape[-1])
    feature_dim = int(features.shape[-1])
    if mechanism == LINEAR_NO_BIAS_POLICY:
        return jnp.zeros((horizon, epsilon_dim, feature_dim), dtype=base_epsilon.dtype)
    if mechanism == AFFINE_POLICY:
        return jnp.zeros((horizon, epsilon_dim, feature_dim + 1), dtype=base_epsilon.dtype)
    raise ValueError(f"unknown mechanism {mechanism!r}")


def _run_adam(
    objective_for_params: Any,
    zero_params: jnp.ndarray,
    *,
    n_steps: int,
    learning_rate: float,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, bool]]:
    tx = optax.adam(float(learning_rate))
    opt_state = tx.init(zero_params)
    params = zero_params
    best_params = zero_params
    best_objective = objective_for_params(zero_params)
    nan_seen = bool(np.isnan(np.asarray(best_objective)))
    inf_seen = bool(np.isinf(np.asarray(best_objective)))

    value_and_grad = jax.value_and_grad(lambda candidate: -objective_for_params(candidate))
    for _ in range(int(n_steps)):
        loss_value, grads = value_and_grad(params)
        objective = -loss_value
        nan_seen = nan_seen or bool(np.isnan(np.asarray(objective)))
        inf_seen = inf_seen or bool(np.isinf(np.asarray(objective)))
        if bool(np.isfinite(np.asarray(objective))) and bool(
            np.asarray(objective > best_objective)
        ):
            best_objective = objective
            best_params = params
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    final_objective = objective_for_params(params)
    if bool(np.isfinite(np.asarray(final_objective))) and bool(
        np.asarray(final_objective > best_objective)
    ):
        best_objective = final_objective
        best_params = params
    nan_seen = nan_seen or bool(np.isnan(np.asarray(final_objective)))
    inf_seen = inf_seen or bool(np.isinf(np.asarray(final_objective)))
    return best_params, best_objective, {"nan_seen": nan_seen, "overflow_seen": inf_seen}


def _directional_hvp(
    task_loss_for_params: Any,
    energy_for_params: Any,
    zero_params: jnp.ndarray,
    grad: jnp.ndarray,
) -> dict[str, Any]:
    grad_norm = jnp.linalg.norm(grad)
    direction = jnp.where(grad_norm > 0, grad / jnp.maximum(grad_norm, 1e-12), grad)
    _, hvp = jax.jvp(jax.grad(task_loss_for_params), (zero_params,), (direction,))
    numerator = float(np.vdot(np.asarray(direction), np.asarray(hvp)))
    denom = float(np.asarray(energy_for_params(direction)))
    ratio = numerator / denom if denom > 0.0 else math.nan
    return directional_curvature_summary(
        hvp_directional_ratio=ratio,
        n_hvp=1,
        method="gradient_direction_jvp",
    )


def _cap_behavior(delta: jnp.ndarray, radius: jnp.ndarray) -> dict[str, Any]:
    norms = _flattened_per_trial_norm(delta).astype(radius.dtype)
    ratio = norms / jnp.maximum(radius, jnp.asarray(1e-12, dtype=radius.dtype))
    boundary = ratio >= jnp.asarray(1.0 - 1e-4, dtype=ratio.dtype)
    return {
        "radius_mean": float(np.mean(np.asarray(radius))),
        "radius_max": float(np.max(np.asarray(radius))),
        "selected_norm_mean": float(np.mean(np.asarray(norms))),
        "selected_norm_max": float(np.max(np.asarray(norms))),
        "selected_norm_radius_ratio_mean": float(np.mean(np.asarray(ratio))),
        "selected_norm_radius_ratio_max": float(np.max(np.asarray(ratio))),
        "boundary_fraction": float(np.mean(np.asarray(boundary, dtype=np.float32))),
        "trust_region": "per_trial_flattened_time_component_l2_cap",
    }


def _replicate_model(model: Any, hps: Any, replicate_index: int) -> Any:
    n_replicates = int(getattr(getattr(hps, "model", hps), "n_replicates", 1))
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_axis_array(leaf, n_replicates),
    )
    replicate_arrays = jt.map(
        lambda leaf: None if leaf is None else leaf[int(replicate_index)],
        model_arrays,
        is_leaf=lambda leaf: leaf is None,
    )
    model_replicate = eqx.combine(replicate_arrays, model_other)
    return _with_single_replicate_state_initializers(
        model_replicate,
        n_replicates=n_replicates,
        replicate_index=int(replicate_index),
    )


def _candidate_lambda(lambda_source: dict[str, Any], *, beta: float) -> float:
    for row in lambda_source["pooled_beta_mapping"]:
        if float(row["beta"]) == float(beta):
            return float(row["lambda"])
    raise ValueError(f"no lambda row for beta={beta}")


def _readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = []
    if any(row["nonfinite"]["nan_seen"] or row["nonfinite"]["overflow_seen"] for row in rows):
        blockers.append("nonfinite objective seen in at least one frozen optimizer row")
    if any(row["accepted_objective_gain"] < -1e-9 for row in rows):
        blockers.append("at least one frozen optimizer row selected an objective below zero")
    if any(not row["batch_size_invariance"]["mean_reduction_invariant"] for row in rows):
        blockers.append("mean-reduced energy failed duplicated-batch invariance")
    return {
        "ready_for_no_launch_spec": not blockers,
        "blockers": blockers,
        "caveat": (
            "Finite production rows now use Adam over finite parameters, but current "
            "training integration materializes static epsilon from a clean rollout pre-step. "
            "Do not describe the next spec as a true live-perturbed closed-loop finite-policy "
            "run until a Feedbax live rollout hook exists."
        ),
    }


def _lambda_recommendations(
    rows: list[dict[str, Any]],
    lambda_input: float,
) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for mechanism in MECHANISMS:
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        pressures = [float(row["gradient_pressure_scale"]) for row in mechanism_rows]
        max_pressure = max(pressures) if pressures else math.nan
        recommendations[mechanism] = {
            "lambda_input": float(lambda_input),
            "max_gradient_pressure_scale": max_pressure,
            "recommended_lambda_floor": max(float(lambda_input), max_pressure),
            "basis": (
                "Use the larger of the pooled beta=1.05 HVP lambda input and the "
                "regenerated frozen-batch gradient-pressure scale across selected "
                "checkpoints. Directional HVP values are retained as diagnostic "
                "curvature sidecars, not as the training lambda gate."
            ),
        }
    return recommendations


def _write_readme(results_dir: Path) -> None:
    readme = results_dir / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "Regenerated frozen-batch gate for issue 9bb676f. Tracked files record "
        "finite-policy Adam routing, checkpoint/batch provenance, and direct/linear/"
        "affine lambda diagnostics; bulk tensors live under `_artifacts/9bb676f/`.\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Frozen Policy Adam Lambda Gate",
        "",
        f"- Source run: `{payload['source_issue']}/{payload['source_run']}`",
        f"- Checkpoints: `{payload['checkpoint_policy']['selected']}`",
        f"- Lambda input: `{payload['lambda_source']['lambda']:.6g}` from "
        f"`{payload['lambda_source']['path']}` beta `{payload['lambda_source']['beta']}`",
        f"- Ready for no-launch spec: `{payload['readiness']['ready_for_no_launch_spec']}`",
        f"- Caveat: {payload['readiness']['caveat']}",
        "",
        "## Lambda Floors",
        "",
        "| mechanism | lambda input | max grad pressure | recommended floor |",
        "|---|---:|---:|---:|",
    ]
    for mechanism, recommendation in payload["mechanism_lambda_recommendations"].items():
        lines.append(
            f"| {mechanism} | {_fmt(recommendation['lambda_input'])} | "
            f"{_fmt(recommendation['max_gradient_pressure_scale'])} | "
            f"{_fmt(recommendation['recommended_lambda_floor'])} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Rows",
            "",
        ]
    )
    lines.extend(
        [
            "| checkpoint | mechanism | grad pressure | directional lambda* | selected energy | objective gain | boundary frac | nonfinite |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for descriptor in payload["descriptors"]:
        checkpoint = descriptor["checkpoint_batches"]
        for row in [
            row
            for row in payload["rows"]
            if f"checkpoint_{int(checkpoint):07d}" in row["tensor_artifact"]
        ]:
            curvature = row["curvature"]
            nonfinite = row["nonfinite"]
            lines.append(
                f"| {checkpoint} | {row['mechanism']} | "
                f"{_fmt(row['gradient_pressure_scale'])} | "
                f"{_fmt(curvature['lambda_star_directional'])} | "
                f"{_fmt(row['selected_energy'])} | "
                f"{_fmt(row['accepted_objective_gain'])} | "
                f"{_fmt(row['cap_behavior']['boundary_fraction'])} | "
                f"nan={nonfinite['nan_seen']}, inf={nonfinite['overflow_seen']} |"
            )
    if payload["readiness"]["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in payload["readiness"]["blockers"])
    update_marked_section(path, "frozen_policy_adam_lambda_gate", "\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _key_list(key: jnp.ndarray) -> list[int]:
    return [int(value) for value in np.asarray(key, dtype=np.uint32).tolist()]


def _repo_relative(path: Path) -> str:
    return str(Path(path).absolute().relative_to(REPO_ROOT.absolute()))


if __name__ == "__main__":
    main()
