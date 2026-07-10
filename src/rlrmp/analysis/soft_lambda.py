"""Shared frozen-batch and CLI helpers for soft-lambda analyses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from feedbax.config.namespace import TreeNamespace
from feedbax.runtime.batch import BatchInfo
from jax_cookbook import load_with_hyperparameters

from rlrmp.train import cs_nominal_gru as nominal
from rlrmp.io import update_marked_section, write_compact_json, write_csv_rows
from rlrmp.train.cs_perturbation_training import (
    BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
    _broad_epsilon_pgd_trust_radius,
    _ensure_broad_epsilon_input,
    _epsilon_time_mask,
    config_from_broad_epsilon_pgd_hps,
)
from rlrmp.train.task_model import setup_task_model_pair


CAP_RADIUS_15CM = 0.004545500088363065
CAP_SOURCE = "ofb_6d_no_integrator_gamma_1p4_rollout_radius"


@dataclass(frozen=True)
class FrozenBatch:
    """Frozen model, trial bank, and active soft-adversary geometry."""

    task: Any
    model: Any
    trial_specs: Any
    keys_model: Any
    hps: TreeNamespace
    run_spec: dict[str, Any]
    radius: jnp.ndarray
    time_mask: jnp.ndarray


def soft_pgd_config(
    *,
    lambda_value: float,
    n_steps: int,
    step_size_fraction: float,
) -> TreeNamespace:
    """Return the canonical moderate soft-energy PGD configuration."""

    return TreeNamespace(
        enabled=True,
        level="moderate",
        budget_scale=1.0,
        reach_length_scaling=False,
        objective={
            "kind": BROAD_EPSILON_PGD_SOFT_ENERGY_OBJECTIVE,
            "lambda": float(lambda_value),
        },
        safety_cap={
            "l2_radius_15cm": CAP_RADIUS_15CM,
            "source": {"key": CAP_SOURCE},
        },
        n_steps=int(n_steps),
        step_size_fraction=float(step_size_fraction),
        epsilon_dim=6,
    )


def load_frozen_batch(
    args: argparse.Namespace,
    run_id: str,
    *,
    repo_root: Path,
) -> FrozenBatch:
    """Load one canonical frozen training batch for a soft-lambda analysis."""

    run_spec_path = repo_root / "results" / args.experiment / "runs" / f"{run_id}.json"
    artifact_dir = repo_root / "_artifacts" / args.experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    parser = nominal.build_parser()
    replay_args = nominal.resolve_run_spec_args(
        parser.parse_args(["--run-spec", str(run_spec_path)]),
        parser=parser,
    )
    hps = nominal.build_hps(replay_args)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model, _ = load_with_hyperparameters(
        artifact_dir / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    model = select_replicate_model(model, hps, int(args.replicate_index))
    batch_size = int(args.batch_size)
    key = jr.fold_in(jr.PRNGKey(seed), stable_run_fold(run_id))
    key_trials, key_model = jr.split(key)
    batch_info = BatchInfo(
        size=batch_size,
        start=0,
        current=0,
        total=int(hps.n_batches_condition),
    )
    trial_specs = eqx.filter_vmap(
        lambda subkey: pair.task.get_train_trial_with_intervenor_params(
            subkey,
            batch_info=batch_info,
        )
    )(jr.split(key_trials, batch_size))
    trial_specs = _ensure_broad_epsilon_input(trial_specs, epsilon_dim=6)
    audit_hps = hps | {
        "broad_epsilon_pgd_training": soft_pgd_config(
            lambda_value=1.0,
            n_steps=1,
            step_size_fraction=0.25,
        )
    }
    cfg = config_from_broad_epsilon_pgd_hps(audit_hps.broad_epsilon_pgd_training)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    radius = _broad_epsilon_pgd_trust_radius(trial_specs, cfg).astype(epsilon.dtype)
    time_mask = _epsilon_time_mask(trial_specs, epsilon, cfg.movement_epoch_only)
    return FrozenBatch(
        task=pair.task,
        model=model,
        trial_specs=trial_specs,
        keys_model=jr.split(key_model, batch_size),
        hps=hps,
        run_spec=run_spec,
        radius=radius,
        time_mask=time_mask,
    )


def select_replicate_model(
    model: Any,
    hps: TreeNamespace,
    replicate_index: int,
) -> Any:
    """Select one replicate while preserving state-index initializers."""

    n_replicates = int(hps.model.n_replicates)
    arrays, other = eqx.partition(
        model,
        lambda leaf: (
            eqx.is_array(leaf)
            and leaf.ndim > 0
            and int(getattr(leaf, "shape", (0,))[0]) == n_replicates
        ),
    )
    selected = jt.map(
        lambda leaf: None if leaf is None else leaf[replicate_index],
        arrays,
        is_leaf=lambda leaf: leaf is None,
    )
    return nominal._with_single_replicate_state_initializers(
        eqx.combine(selected, other),
        n_replicates=n_replicates,
        replicate_index=replicate_index,
    )


def stable_run_fold(run_id: str) -> int:
    """Return the stable integer fold used for frozen-batch PRNG keys."""

    return sum((index + 1) * ord(char) for index, char in enumerate(run_id))


def base_parser(
    *,
    description: str | None,
    experiment: str,
    issue: str,
    batch_size: int,
    replicate_index: int = 0,
) -> argparse.ArgumentParser:
    """Return the shared soft-lambda materializer parser prefix."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment", default=experiment)
    parser.add_argument("--issue", default=issue)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--replicate-index", type=int, default=replicate_index)
    return parser


def materialize_write_print(
    *,
    materialize: Callable[[], dict[str, Any]],
    writers: Sequence[Callable[[dict[str, Any]], None]],
    summarize: Callable[[dict[str, Any]], Any],
    printer: Callable[[Any], None],
) -> None:
    """Materialize once, run ordered writers, and print a compact summary."""

    payload = materialize()
    for writer in writers:
        writer(payload)
    printer(summarize(payload))


def run_soft_lambda_materializer(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    materialize: Callable[[argparse.Namespace], dict[str, Any]],
    csv_rows: Callable[[dict[str, Any]], Sequence[Mapping[str, Any]]],
    csv_fields: Sequence[str],
    render_markdown: Callable[[dict[str, Any]], str],
    marker: str,
    json_indent: int | None = None,
    json_writer: Callable[[Path, dict[str, Any]], Mapping[str, Any] | None] | None = None,
    extra_summary: Callable[[dict[str, Any]], Mapping[str, Any]] = lambda _payload: {},
) -> int:
    """Run the shared materialize -> write outputs -> print-manifest CLI contract."""

    payload = materialize(args)
    output_json = repo_root / args.output_json
    output_csv = repo_root / args.output_csv
    output_md = repo_root / args.output_md
    if json_writer is None:
        if json_indent is None:
            write_compact_json(output_json, payload)
        else:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(payload, indent=json_indent, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        json_metadata: Mapping[str, Any] = {}
    else:
        json_metadata = json_writer(output_json, payload) or {}
    write_csv_rows(output_csv, csv_rows(payload), fieldnames=csv_fields)
    update_marked_section(output_md, marker, render_markdown(payload))
    summary = {
        "json": str(output_json),
        **json_metadata,
        "csv": str(output_csv),
        "markdown": str(output_md),
        **extra_summary(payload),
    }
    print(json.dumps(summary, indent=2))
    return 0


def render_soft_lambda_redo_markdown(
    payload: Mapping[str, Any],
    *,
    title: str,
    introduction: str,
    objective_title: str,
    objective_header: str,
    objective_alignment: str,
    objective_rows: Iterable[str],
    extra_sections: Iterable[str] = (),
    footer: Iterable[str] = (),
) -> str:
    """Render the canonical HVP/beta report with analysis-specific tables."""

    source = payload["hvp_source"]
    lines = f"""# {title}

Issue: `{payload['issue']}`. Source no-PGD substrates: `c92ebd8`.

{introduction}

## Source contract

HVP source: `{source['path']}` (`{source['schema_version']}`). Primary scale: `{source['primary_continuity_summary']}`.

Beta mapping: `lambda(beta) = beta^2 * substrate_p90(lambda_star_i)`. Beta `0.95` is diagnostic only. Cap/interiority is not used as a criterion; old-cap ratios below are sidecars only.

## HVP/p90 beta mapping

| substrate | beta | role | lambda_star p90 | lambda | source |
|---|---:|---|---:|---:|---|""".splitlines()
    for run in payload["rows"]:
        for mapping in run["beta_mapping"]:
            lines.append(
                f"| `{run['run_id']}` | {mapping['beta']:.3g} | {mapping['role']} | "
                f"{mapping['lambda_star_summary_value']:.6g} | {mapping['lambda']:.6g} | "
                f"{mapping['lambda_source']} |"
            )
    lines.extend(["", f"## {objective_title}", "", objective_header, objective_alignment])
    lines.extend(objective_rows)
    lines.extend(extra_sections)
    lines.extend("\n## Classification counts\n\n| substrate | counts |\n|---|---|".splitlines())
    for run in payload["rows"]:
        counts = ", ".join(
            f"`{label}`: {count}" for label, count in run["objective_classification_counts"].items()
        )
        lines.append(f"| `{run['run_id']}` | {counts} |")
    lines.extend(["", *footer])
    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines)
