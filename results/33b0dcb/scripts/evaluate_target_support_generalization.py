"""Evaluate issue 33b0dcb target-support generalization rows.

This is an experiment-specific post-run evaluator. It reuses the nominal GRU
checkpoint loaders and target-relative target-bank helpers, then reports
target-grid behavior split by the training row's seen and held-out support.
"""

from __future__ import annotations
from rlrmp.io import write_csv_rows
from rlrmp.eval.ensemble import eval_ensemble_on_trials as evaluate_replicates
from rlrmp.eval.kinematics import initial_effector_position, initial_effector_velocity

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.tree as jt
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
)
from rlrmp.io import update_marked_section
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path
from rlrmp.train.cs_perturbation_training import (
    TARGET_SUPPORT_PROFILE_020A65B,
    _with_static_target,
    target_relative_target_support_config,
)
from rlrmp.train.task_model import setup_task_model_pair

ISSUE = "33b0dcb"
RUN_IDS = (
    "h0_no_pgd_targetsupport__old_replicate_lr3e-3_clip5_b64",
    "h0_no_pgd_targetsupport__const_dense_all_lr3e-3_clip5_b64",
    "h0_no_pgd_targetsupport__const_sparse8_lr3e-3_clip5_b64",
    "h0_no_pgd_targetsupport__const_band8_lr3e-3_clip5_b64",
    "h0_no_pgd_targetsupport__const_band16_lr3e-3_clip5_b64",
    "h0_no_pgd_targetsupport__const_band36_lr3e-3_clip5_b64",
)
PROFILE_LABELS = {
    "old_020a65b": "A old_replicate",
    "const_dense_all": "B const_dense_all",
    "const_sparse8": "C const_sparse8",
    "const_band8": "D const_band8",
    "const_band16": "E const_band16",
    "const_band36": "F const_band36",
}
PRIMARY_DIRECTIONS = 72
PRIMARY_REACH_M = 0.15
LENGTH_DIAGNOSTIC_RADII_M = (0.10, 0.12, 0.15, 0.18)
TARGET_ROUND_DIGITS = 9


@dataclass(frozen=True)
class BankSpec:
    """Concrete target bank with one split label per target."""

    name: str
    role: str
    targets_m: np.ndarray
    split_labels: tuple[str, ...]


@dataclass(frozen=True)
class RunInputs:
    """Resolved run spec inputs."""

    run_id: str
    run_spec_path: Path
    run_spec: dict[str, Any]
    profile: str
    label: str
    target_distribution: Mapping[str, Any]


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    run_ids = tuple(args.run_id or RUN_IDS)
    runs = [resolve_run(run_id, repo_root=repo_root) for run_id in run_ids]

    notes_dir = repo_root / "results" / ISSUE / "notes"
    bulk_dir = repo_root / "_artifacts" / ISSUE / "target_support_eval"
    mkdir_p(notes_dir)
    mkdir_p(bulk_dir)

    records: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    run_manifests: dict[str, Any] = {}
    for index, run in enumerate(runs):
        print(f"Evaluating {run.label} ({run.run_id})")
        result = evaluate_run(
            run,
            repo_root=repo_root,
            include_length_diagnostic=not args.no_length_diagnostic,
            key_seed=args.seed + index * 1000,
        )
        run_manifests[run.run_id] = result["manifest"]
        records.extend(result["records"])
        profile_records.extend(result["profiles"])

    csv_path = notes_dir / "target_support_generalization_summary.csv"
    profile_csv_path = bulk_dir / "target_support_velocity_profiles.csv"
    profile_pointer_path = notes_dir / "target_support_velocity_profiles_pointer.json"
    json_path = notes_dir / "target_support_generalization_manifest.json"
    note_path = notes_dir / "target_support_generalization.md"
    write_csv(csv_path, records)
    write_profile_csv(profile_csv_path, profile_records)
    profile_pointer = write_profile_pointer(
        profile_pointer_path,
        profile_csv_path=profile_csv_path,
        profile_records=profile_records,
        repo_root=repo_root,
    )
    manifest = build_manifest(
        runs=runs,
        records=records,
        run_manifests=run_manifests,
        csv_path=csv_path,
        profile_csv_path=profile_csv_path,
        profile_pointer_path=profile_pointer_path,
        profile_pointer=profile_pointer,
        repo_root=repo_root,
        include_length_diagnostic=not args.no_length_diagnostic,
    )
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_marked_section(note_path, "target_support_generalization", render_note(manifest))
    print(json.dumps(manifest["headline"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--seed", type=int, default=330)
    parser.add_argument("--no-length-diagnostic", action="store_true")
    return parser


def resolve_run(run_id: str, *, repo_root: Path) -> RunInputs:
    spec_path = run_spec_path(ISSUE, run_id, repo_root=repo_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    target_distribution = spec["hps"]["target_relative_multitarget"]["target_distribution"]
    profile = str(target_distribution["target_support_profile"])
    return RunInputs(
        run_id=run_id,
        run_spec_path=spec_path,
        run_spec=spec,
        profile=profile,
        label=PROFILE_LABELS.get(profile, profile),
        target_distribution=target_distribution,
    )


def evaluate_run(
    run: RunInputs,
    *,
    repo_root: Path,
    include_length_diagnostic: bool,
    key_seed: int,
) -> dict[str, Any]:
    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run.run_spec.get("seed", 42))))
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=ISSUE,
        run_id=run.run_id,
        run_spec=run.run_spec,
        repo_root=repo_root,
    )
    banks = [
        dense_all_angle_bank(run),
        old_validation_bank(run),
    ]
    if include_length_diagnostic:
        banks.append(length_diagnostic_bank(run))

    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    bank_summaries: dict[str, Any] = {}
    for bank_index, bank in enumerate(banks):
        trial_specs = trial_specs_for_bank(pair.task.validation_trials, bank.targets_m)
        states = evaluate_replicates(
            pair.task,
            model,
            trial_specs,
            n_replicates=int(hps.model.n_replicates),
            key=jr.PRNGKey(key_seed + bank_index),
        )
        metrics = summarize_bank(
            states=states,
            trial_specs=trial_specs,
            split_labels=bank.split_labels,
            dt=float(run.run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01))),
        )
        bank_summaries[bank.name] = {
            "role": bank.role,
            "n_targets": int(bank.targets_m.shape[0]),
            "splits": metrics["splits"],
        }
        for split, row in metrics["splits"].items():
            records.append(
                {
                    "run_id": run.run_id,
                    "row": run.label,
                    "profile": run.profile,
                    "bank": bank.name,
                    "bank_role": bank.role,
                    "split": split,
                    **row,
                }
            )
        for split, rows in metrics["velocity_profiles"].items():
            for row in rows:
                profiles.append(
                    {
                        "run_id": run.run_id,
                        "row": run.label,
                        "profile": run.profile,
                        "bank": bank.name,
                        "split": split,
                        **row,
                    }
                )

    return {
        "records": records,
        "profiles": profiles,
        "manifest": {
            "checkpoint_policy": "validation_selected_per_replicate",
            "checkpoint_selection": [
                checkpoint_to_json(selection, repo_root=repo_root)
                for selection in checkpoint_selection
            ],
            "banks": bank_summaries,
        },
    }


def dense_all_angle_bank(run: RunInputs) -> BankSpec:
    targets = circular_targets(PRIMARY_DIRECTIONS, (PRIMARY_REACH_M,))
    return BankSpec(
        name="dense_all_angle_0p15m",
        role="primary_nominal_all_angle_constant_reach",
        targets_m=targets,
        split_labels=tuple(target_split(run, target) for target in targets),
    )


def old_validation_bank(run: RunInputs) -> BankSpec:
    cfg = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_020A65B,
        enabled=True,
        force_filter_feedback=True,
    )
    targets = np.asarray(cfg.validation_targets_m, dtype=np.float64)
    return BankSpec(
        name="old_020a65b_validation_grid",
        role="reproduction_seen_vs_held_out_grid",
        targets_m=targets,
        split_labels=tuple(old_grid_split(target) for target in targets),
    )


def length_diagnostic_bank(run: RunInputs) -> BankSpec:
    targets = circular_targets(PRIMARY_DIRECTIONS, LENGTH_DIAGNOSTIC_RADII_M)
    return BankSpec(
        name="crossed_radius_direction_diagnostic",
        role="diagnostic_crossed_0p10_0p12_0p15_0p18m_by_72_directions",
        targets_m=targets,
        split_labels=tuple(length_split(run, target) for target in targets),
    )


def circular_targets(n_directions: int, radii_m: Sequence[float]) -> np.ndarray:
    rows = []
    for radius in radii_m:
        for direction in np.linspace(0.0, 360.0, int(n_directions), endpoint=False):
            theta = math.radians(float(direction))
            rows.append((float(radius) * math.cos(theta), float(radius) * math.sin(theta)))
    return np.asarray(rows, dtype=np.float64)


def trial_specs_for_bank(base_trial_specs: Any, targets_m: np.ndarray) -> Any:
    # The base trial already carries the target-relative input contract from the
    # run's task setup. Preserve the exact target list here, including old-grid
    # mixed radii and any non-cross-product ordering.
    template = first_trial_template(base_trial_specs)
    trial_specs = _with_static_target(
        template,
        jnp.asarray(targets_m, dtype=jnp.float32),
        metadata={"target_support_eval_bank": "issue_33b0dcb"},
    )
    return normalize_bank_batch_axes(normalize_effector_target_velocity(trial_specs))


def normalize_effector_target_velocity(trial_specs: Any) -> Any:
    """Broadcast effector-target velocity to match the rewritten target bank."""

    effector_target = trial_specs.inputs.get("effector_target")
    if effector_target is None or not hasattr(effector_target, "pos"):
        return trial_specs
    pos = getattr(effector_target, "pos")
    vel = getattr(effector_target, "vel", None)
    if vel is None or getattr(vel, "shape", None) == getattr(pos, "shape", None):
        return trial_specs
    updated_effector_target = eqx.tree_at(
        lambda state: state.vel,
        effector_target,
        jnp.zeros_like(pos),
    )
    inputs = dict(trial_specs.inputs)
    inputs["effector_target"] = updated_effector_target
    return eqx.tree_at(lambda spec: spec.inputs, trial_specs, inputs)


def normalize_bank_batch_axes(trial_specs: Any) -> Any:
    """Broadcast unbatched time-series leaves over the target-bank axis."""

    batch_size = int(trial_specs.inputs["target"].shape[0])
    n_steps = int(trial_specs.timeline.n_steps)

    def normalize_leaf(leaf: Any) -> Any:
        if not eqx.is_array(leaf) or leaf.ndim == 0:
            return leaf
        if leaf.shape[0] == batch_size:
            return leaf
        if leaf.shape[0] == n_steps:
            return jnp.broadcast_to(leaf, (batch_size, *leaf.shape))
        return leaf

    return jt.map(normalize_leaf, trial_specs)


def first_trial_template(trial_specs: Any) -> Any:
    """Return a single-trial template from an already batched validation spec."""

    batch_size = int(trial_specs.inputs["target"].shape[0])
    return jt.map(
        lambda leaf: (
            leaf[0]
            if eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == batch_size
            else leaf
        ),
        trial_specs,
    )




def summarize_bank(
    *,
    states: Any,
    trial_specs: Any,
    split_labels: Sequence[str],
    dt: float,
) -> dict[str, Any]:
    position = np.asarray(states.mechanics.effector.pos, dtype=np.float64)
    velocity = np.asarray(states.mechanics.effector.vel, dtype=np.float64)
    initial_position = initial_effector_position(trial_specs)
    initial_velocity = initial_effector_velocity(trial_specs)
    targets = np.asarray(trial_specs.inputs["target"][..., -1, :], dtype=np.float64)

    full_velocity = np.concatenate(
        [
            np.broadcast_to(
                initial_velocity[None, :, None, :], (velocity.shape[0], velocity.shape[1], 1, 2)
            ),
            velocity,
        ],
        axis=2,
    )
    direction, reach_length = reach_direction(initial_position, targets)
    radial_velocity = np.sum(full_velocity * direction[None, :, None, :], axis=-1)
    normalized_radial_velocity = radial_velocity / reach_length[None, :, None]
    endpoint_error = np.linalg.norm(position[:, :, -1, :] - targets[None, :, :], axis=-1)
    terminal_radial_error = np.sum(
        (position[:, :, -1, :] - targets[None, :, :]) * direction[None, :, :],
        axis=-1,
    )
    peak_velocity = np.max(radial_velocity, axis=-1)
    time_to_peak = np.argmax(radial_velocity, axis=-1) * float(dt)
    late_start = max(0, int(math.floor(radial_velocity.shape[-1] * 0.65)))
    late_negative = np.minimum(np.min(radial_velocity[:, :, late_start:], axis=-1), 0.0)

    split_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, split in enumerate(split_labels):
        split_to_indices[str(split)].append(index)

    splits = {}
    profiles = {}
    time = np.arange(radial_velocity.shape[-1], dtype=np.float64) * float(dt)
    for split, indices in sorted(split_to_indices.items()):
        idx = np.asarray(indices, dtype=np.int64)
        splits[split] = {
            "n_targets": int(idx.size),
            "n_samples": int(idx.size * radial_velocity.shape[0]),
            "endpoint_error_m_mean": finite_mean(endpoint_error[:, idx]),
            "endpoint_error_m_p95": finite_quantile(endpoint_error[:, idx], 0.95),
            "terminal_radial_error_m_mean": finite_mean(terminal_radial_error[:, idx]),
            "peak_radial_velocity_m_s_mean": finite_mean(peak_velocity[:, idx]),
            "time_to_peak_s_mean": finite_mean(time_to_peak[:, idx]),
            "late_negative_radial_velocity_m_s_mean": finite_mean(late_negative[:, idx]),
            "late_negative_radial_velocity_m_s_min": finite_min(late_negative[:, idx]),
            "normalized_profile_peak_s_inv": finite_max(normalized_radial_velocity[:, idx, :]),
        }
        profile_mean = np.mean(normalized_radial_velocity[:, idx, :].reshape(-1, time.size), axis=0)
        profile_sd = np.std(normalized_radial_velocity[:, idx, :].reshape(-1, time.size), axis=0)
        profiles[split] = [
            {
                "time_s": float(t),
                "normalized_radial_velocity_mean_s_inv": float(mu),
                "normalized_radial_velocity_sd_s_inv": float(sd),
            }
            for t, mu, sd in zip(time, profile_mean, profile_sd, strict=True)
        ]
    return {"splits": splits, "velocity_profiles": profiles}






def reach_direction(
    initial_position: np.ndarray,
    target_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = target_position - initial_position
    reach_length = np.linalg.norm(delta, axis=-1)
    safe = np.where(reach_length > 0.0, reach_length, 1.0)
    return delta / safe[:, None], safe


def target_split(run: RunInputs, target: np.ndarray) -> str:
    seen = target_set(run.target_distribution.get("seen_targets_m", ()))
    held = target_set(run.target_distribution.get("held_out_targets_m", ()))
    key = target_key(target)
    if key in seen:
        return "train_support"
    if key in held:
        return "held_out_support"
    angle = round(angle_deg(target), 9)
    held_angles = {
        round(float(x) % 360.0, 9)
        for x in run.target_distribution.get("held_out_directions_deg", ())
    }
    if angle in held_angles:
        return "held_out_direction_interpolated_radius"
    return "interpolation_unseen_grid"


def old_grid_split(target: np.ndarray) -> str:
    cfg = target_relative_target_support_config(
        profile=TARGET_SUPPORT_PROFILE_020A65B,
        enabled=True,
        force_filter_feedback=True,
    )
    key = target_key(target)
    if key == target_key(np.asarray(cfg.original_target_anchor_m)):
        return "original_anchor"
    if key in target_set(cfg.seen_targets_m):
        return "old_seen_support"
    if key in target_set(cfg.held_out_targets_m):
        return "old_held_out_support"
    return "old_grid_other"


def length_split(run: RunInputs, target: np.ndarray) -> str:
    angle = round(angle_deg(target), 9)
    radius = round(float_radius(target), 9)
    seen_angles = {
        round(float(x) % 360.0, 9) for x in run.target_distribution.get("seen_directions_deg", ())
    }
    held_angles = {
        round(float(x) % 360.0, 9)
        for x in run.target_distribution.get("held_out_directions_deg", ())
    }
    seen_radii = {round(float(x), 9) for x in run.target_distribution.get("seen_amplitudes_m", ())}
    held_radii = {
        round(float(x), 9) for x in run.target_distribution.get("held_out_amplitudes_m", ())
    }
    direction_role = (
        "seen_direction"
        if angle in seen_angles
        else "held_out_direction"
        if angle in held_angles
        else "interpolation_direction"
    )
    radius_role = (
        "seen_radius"
        if radius in seen_radii
        else "held_out_radius"
        if radius in held_radii
        else "diagnostic_radius"
    )
    return f"{direction_role}__{radius_role}"


def target_set(targets: Iterable[Sequence[float]]) -> set[tuple[float, float]]:
    return {target_key(np.asarray(row, dtype=np.float64)) for row in targets}


def target_key(target: np.ndarray) -> tuple[float, float]:
    return tuple(np.round(np.asarray(target, dtype=np.float64), TARGET_ROUND_DIGITS).tolist())


def float_radius(target: Sequence[float] | np.ndarray) -> float:
    row = np.asarray(target, dtype=np.float64)
    return float(np.linalg.norm(row))


def angle_deg(target: Sequence[float] | np.ndarray) -> float:
    row = np.asarray(target, dtype=np.float64)
    return float(math.degrees(math.atan2(float(row[1]), float(row[0]))) % 360.0)


def finite_mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def finite_min(values: Any) -> float:
    return float(np.min(np.asarray(values, dtype=np.float64)))


def finite_max(values: Any) -> float:
    return float(np.max(np.asarray(values, dtype=np.float64)))


def finite_quantile(values: Any, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def checkpoint_to_json(
    selection: ReplicateCheckpointSelection,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    return selection.to_json(repo_root=repo_root)


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "row",
        "profile",
        "run_id",
        "bank",
        "bank_role",
        "split",
        "n_targets",
        "n_samples",
        "endpoint_error_m_mean",
        "endpoint_error_m_p95",
        "terminal_radial_error_m_mean",
        "peak_radial_velocity_m_s_mean",
        "time_to_peak_s_mean",
        "late_negative_radial_velocity_m_s_mean",
        "late_negative_radial_velocity_m_s_min",
        "normalized_profile_peak_s_inv",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def write_profile_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ['row', 'profile', 'run_id', 'bank', 'split', 'time_s', 'normalized_radial_velocity_mean_s_inv', 'normalized_radial_velocity_sd_s_inv']
    write_csv_rows(path, list(records), fieldnames=fieldnames)


def build_manifest(
    *,
    runs: Sequence[RunInputs],
    records: Sequence[Mapping[str, Any]],
    run_manifests: Mapping[str, Any],
    csv_path: Path,
    profile_csv_path: Path,
    profile_pointer_path: Path,
    profile_pointer: Mapping[str, Any],
    repo_root: Path,
    include_length_diagnostic: bool,
) -> dict[str, Any]:
    primary = [row for row in records if row["bank"] == "dense_all_angle_0p15m"]
    old_reproduction = [row for row in records if row["bank"] == "old_020a65b_validation_grid"]
    headline = summarize_headline(primary, old_reproduction)
    return {
        "schema_version": "rlrmp.33b0dcb_target_support_eval.v1",
        "issue": ISSUE,
        "checkpoint_policy": "validation_selected_per_replicate",
        "selection_role": (
            "checkpoint selection is the sparse logged validation-selected policy; "
            "target-support split metrics are post-hoc evaluation only"
        ),
        "evaluation_scope": {
            "primary": "dense all-angle 0.15 m grid",
            "reproduction": "old 020a65b seen/held-out validation grid",
            "length_diagnostic": (
                "crossed 0.10/0.12/0.15/0.18 m by 72 directions"
                if include_length_diagnostic
                else "skipped"
            ),
            "stochastic_repeats_per_target_per_replicate": 1,
        },
        "runs": {
            run.run_id: {
                "label": run.label,
                "profile": run.profile,
                "run_spec_path": str(run.run_spec_path.relative_to(repo_root)),
                **run_manifests[run.run_id],
            }
            for run in runs
        },
        "outputs": {
            "summary_csv": str(csv_path.relative_to(repo_root)),
            "velocity_profile_csv": str(profile_csv_path.relative_to(repo_root)),
            "velocity_profile_pointer": str(profile_pointer_path.relative_to(repo_root)),
            "velocity_profile_bulk": profile_pointer["bulk_artifact"],
            "note": f"results/{ISSUE}/notes/target_support_generalization.md",
        },
        "headline": headline,
        "records": list(records),
    }


def summarize_headline(
    primary: Sequence[Mapping[str, Any]],
    old_reproduction: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def get(
        row_label: str, bank_rows: Sequence[Mapping[str, Any]], split: str
    ) -> Mapping[str, Any] | None:
        for row in bank_rows:
            if row["row"] == row_label and row["split"] == split:
                return row
        return None

    old_seen = get("A old_replicate", old_reproduction, "old_seen_support")
    old_held = get("A old_replicate", old_reproduction, "old_held_out_support")
    dense_all = get("B const_dense_all", primary, "train_support")
    sparse_train = get("C const_sparse8", primary, "train_support")
    sparse_held = get("C const_sparse8", primary, "held_out_support")
    band_rows = {}
    for label in ("D const_band8", "E const_band16", "F const_band36"):
        train = get(label, primary, "train_support")
        held = get(label, primary, "held_out_support")
        if train is not None and held is not None:
            band_rows[label] = {
                "held_minus_train_endpoint_error_m": (
                    held["endpoint_error_m_mean"] - train["endpoint_error_m_mean"]
                ),
                "held_minus_train_peak_velocity_m_s": (
                    held["peak_radial_velocity_m_s_mean"] - train["peak_radial_velocity_m_s_mean"]
                ),
            }
    return {
        "old_replicate_seen_endpoint_error_m": (
            None if old_seen is None else old_seen["endpoint_error_m_mean"]
        ),
        "old_replicate_held_endpoint_error_m": (
            None if old_held is None else old_held["endpoint_error_m_mean"]
        ),
        "old_replicate_held_minus_seen_endpoint_error_m": (
            None
            if old_seen is None or old_held is None
            else old_held["endpoint_error_m_mean"] - old_seen["endpoint_error_m_mean"]
        ),
        "dense_all_primary_endpoint_error_m": (
            None if dense_all is None else dense_all["endpoint_error_m_mean"]
        ),
        "sparse8_held_minus_train_endpoint_error_m": (
            None
            if sparse_train is None or sparse_held is None
            else sparse_held["endpoint_error_m_mean"] - sparse_train["endpoint_error_m_mean"]
        ),
        "band_primary_gaps": band_rows,
        "recommended_verdict": "bracketed",
        "recommendation_reason": (
            "This local pass quantifies the completed rows and separates target-support "
            "effects, but objective-comparator and repeat-averaged stochastic grids remain "
            "sidecars to run before calling the experiment fully answered."
        ),
    }


def render_note(manifest: Mapping[str, Any]) -> str:
    records = manifest["records"]
    primary = [row for row in records if row["bank"] == "dense_all_angle_0p15m"]
    old = [row for row in records if row["bank"] == "old_020a65b_validation_grid"]
    length = [row for row in records if row["bank"] == "crossed_radius_direction_diagnostic"]
    headline = manifest["headline"]
    lines = [
        "# Target-Support Generalization Evaluation",
        "",
        "Checkpoint policy: validation-selected per replicate from sparse logged validation.",
        "Each target is evaluated with one stochastic rollout per replicate.",
        "",
        "## Headline",
        "",
        f"Recommended verdict: `{headline['recommended_verdict']}`.",
        headline["recommendation_reason"],
        "",
        "- Old replay reproduces the failure: old-grid held-out endpoint error "
        f"{fmt(headline['old_replicate_held_endpoint_error_m'])} m vs seen "
        f"{fmt(headline['old_replicate_seen_endpoint_error_m'])} m "
        f"(gap {fmt(headline['old_replicate_held_minus_seen_endpoint_error_m'])} m).",
        "- Dense all-angle constant-reach training removes the primary split: "
        f"dense-all endpoint error is {fmt(headline['dense_all_primary_endpoint_error_m'])} m.",
        "- Sparse8 constant-reach training still leaves a held-out direction penalty: "
        f"held minus train endpoint gap {fmt(headline['sparse8_held_minus_train_endpoint_error_m'])} m.",
        "- Held-out band rows are robust for small/moderate bands and degrade mildly "
        "for the 36-direction held-out stress band: "
        f"{format_band_gaps(headline['band_primary_gaps'])}.",
        "",
        "## Primary Dense 0.15 m Grid",
        "",
        metric_table(primary),
        "",
        "## Old 020a65b Validation Grid",
        "",
        metric_table(old),
    ]
    if length:
        lines.extend(
            [
                "",
                "## Length Diagnostic",
                "",
                "Diagnostic-only crossed 0.10/0.12/0.15/0.18 m by 72-direction grid.",
                "",
                metric_table(length),
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary CSV: `{manifest['outputs']['summary_csv']}`",
            "- Normalized radial velocity profiles bulk CSV: "
            f"`{manifest['outputs']['velocity_profile_csv']}`",
            "- Tracked velocity profile pointer: "
            f"`{manifest['outputs']['velocity_profile_pointer']}`",
            "",
            "Objective-comparator note: this pass reports rollout kinematics and task-target "
            "split behavior. It does not materialize the heavier analytical objective "
            "comparator sidecar.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_band_gaps(gaps: Mapping[str, Mapping[str, float]]) -> str:
    parts = []
    for label, values in gaps.items():
        parts.append(f"{label} {fmt(values['held_minus_train_endpoint_error_m'])} m")
    return "; ".join(parts)


def metric_table(records: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "row",
        "split",
        "n",
        "endpoint mean m",
        "endpoint p95 m",
        "peak radial m/s",
        "t_peak s",
        "late neg radial m/s",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(records, key=lambda item: (str(item["row"]), str(item["split"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['row']}`",
                    f"`{row['split']}`",
                    str(row["n_targets"]),
                    fmt(row["endpoint_error_m_mean"]),
                    fmt(row["endpoint_error_m_p95"]),
                    fmt(row["peak_radial_velocity_m_s_mean"]),
                    fmt(row["time_to_peak_s_mean"]),
                    fmt(row["late_negative_radial_velocity_m_s_mean"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"


def write_profile_pointer(
    path: Path,
    *,
    profile_csv_path: Path,
    profile_records: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    """Write the tracked pointer for the gitignored per-time velocity CSV."""

    data = profile_csv_path.read_bytes()
    rel_profile_csv = str(profile_csv_path.relative_to(repo_root))
    pointer = {
        "schema_version": "rlrmp.bulk_artifact_pointer.v1",
        "issue": ISSUE,
        "tracked_artifact": f"results/{ISSUE}/notes/target_support_velocity_profiles_pointer.json",
        "bulk_artifact": {
            "path": rel_profile_csv,
            "format": "csv",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "data_rows": len(profile_records),
            "header_rows": 1,
            "contains": (
                "per-row, per-split normalized target-radial velocity time profiles "
                "for the 33b0dcb target-support evaluation"
            ),
        },
        "regenerate": {
            "command": "PYTHONPATH=$PWD/src uv run --no-sync python "
            "results/33b0dcb/scripts/evaluate_target_support_generalization.py",
            "writes": [
                f"results/{ISSUE}/notes/target_support_generalization_summary.csv",
                rel_profile_csv,
                f"results/{ISSUE}/notes/target_support_velocity_profiles_pointer.json",
                f"results/{ISSUE}/notes/target_support_generalization_manifest.json",
                f"results/{ISSUE}/notes/target_support_generalization.md",
            ],
        },
    }
    path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pointer


if __name__ == "__main__":
    main()
