"""Backfill regeneration specs for active GRU diagnostics.

This script is intentionally conservative. It creates post-hoc provenance specs
for recent GRU diagnostics whose tracked manifests already exist, without
recomputing diagnostics or touching ignored bulk arrays. Historical bridge
manifests, smoke outputs, and legacy run/config specs are out of scope.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.paths import REPO_ROOT


DEFAULT_EXPERIMENTS = (
    "3e66604",
    "5f70333",
    "aacb9ed",
    "ba82f3d",
    "643f101",
    "0203d1f",
    "1ad3c16",
    "3992394",
    "57ab156",
)
TRACKING_ISSUE = "25c4c36"
BACKFILL_SCHEMA_VERSION = "rlrmp.diagnostic_regeneration_backfill_index.v1"


def main() -> None:
    args = build_parser().parse_args()
    created = backfill_experiments(
        experiments=tuple(args.experiment),
        repo_root=args.repo_root,
        dry_run=args.dry_run,
    )
    print(json.dumps({"created_or_updated_specs": created}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        default=list(DEFAULT_EXPERIMENTS),
        help="Experiment/issue ID to backfill. Defaults to the recent active GRU set.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def backfill_experiments(
    *,
    experiments: Sequence[str],
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
) -> list[str]:
    """Backfill regeneration specs for the approved active GRU experiment set."""

    created: list[str] = []
    for experiment in experiments:
        if experiment == "1ad3c16":
            created.extend(backfill_perturbation_calibration(repo_root=repo_root, dry_run=dry_run))
            continue
        created.extend(backfill_gru_experiment(experiment, repo_root=repo_root, dry_run=dry_run))
    return sorted(dict.fromkeys(created))


def backfill_gru_experiment(
    experiment: str,
    *,
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
) -> list[str]:
    """Backfill one GRU experiment notes directory."""

    notes_dir = repo_root / "results" / experiment / "notes"
    if not notes_dir.exists():
        return []
    created: list[str] = []
    for manifest_path in sorted(notes_dir.glob("*.json")):
        if not _should_backfill_manifest(manifest_path):
            continue
        manifest = _read_json(manifest_path)
        classification = classify_manifest(manifest_path.name, manifest)
        if classification is None:
            continue
        spec_path = regeneration_spec_path(manifest_path)
        if not dry_run:
            _add_regeneration_pointer(manifest_path, spec_path, repo_root=repo_root)
            write_regeneration_spec(
                spec_path=spec_path,
                diagnostic_name=classification["diagnostic_name"],
                materializer=classification["materializer"],
                command=None,
                parameters={
                    "experiment": experiment,
                    "manifest": repo_relative(manifest_path, repo_root=repo_root),
                    "schema_or_format": manifest.get("schema_version") or manifest.get("format"),
                    "posthoc_backfill": True,
                    "tracking_issue": TRACKING_ISSUE,
                }
                | classification.get("parameters", {}),
                inputs=inputs_for_manifest(
                    experiment,
                    manifest,
                    manifest_path=manifest_path,
                    repo_root=repo_root,
                ),
                outputs=outputs_for_manifest(
                    experiment,
                    manifest,
                    manifest_path=manifest_path,
                    classification=classification,
                    repo_root=repo_root,
                ),
                source_files=classification["source_files"],
                notes=[
                    "Post-hoc regeneration spec; no diagnostic metrics were recomputed.",
                    "Historical exact commit may differ; use this as the intended rerun path under current GRU diagnostic machinery.",
                ],
                repo_root=repo_root,
            )
        created.append(repo_relative(spec_path, repo_root=repo_root))
    if created and not dry_run:
        index_path = notes_dir / f"diagnostic_regeneration_specs_{experiment}_posthoc.json"
        _write_index(
            index_path,
            experiment=experiment,
            created=created,
            repo_root=repo_root,
        )
        created.append(repo_relative(index_path, repo_root=repo_root))
    return created


def backfill_perturbation_calibration(
    *,
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
) -> list[str]:
    """Backfill the perturbation calibration artifact/note pair."""

    experiment = "1ad3c16"
    output_path = (
        repo_root
        / "_artifacts"
        / experiment
        / "perturbation_open_loop_calibration"
        / "perturbation_open_loop_calibration.json"
    )
    note_path = (
        repo_root / "results" / experiment / "notes" / "perturbation_open_loop_calibration.md"
    )
    spec_path = (
        repo_root
        / "results"
        / experiment
        / "notes"
        / "perturbation_open_loop_calibration_regeneration_spec.json"
    )
    if not output_path.exists() and not note_path.exists():
        return []
    if not dry_run:
        write_regeneration_spec(
            spec_path=spec_path,
            diagnostic_name="perturbation_open_loop_calibration",
            materializer="rlrmp.analysis.pipelines.gru_perturbation_calibration.materialize_perturbation_open_loop_calibration",
            command=None,
            parameters={
                "experiment": experiment,
                "posthoc_backfill": True,
                "tracking_issue": TRACKING_ISSUE,
            },
            inputs=[
                {
                    "role": "declared_source_model",
                    "path": "results/5f70333/runs/lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64/run.json",
                }
            ],
            outputs=[
                {"role": "calibration_bulk_manifest", "path": output_path},
                {"role": "calibration_note", "path": note_path},
            ],
            source_files=[
                "src/rlrmp/analysis/pipelines/gru_perturbation_calibration.py",
                "src/rlrmp/eval/perturbation_bank.py",
            ],
            notes=[
                "Backfilled for the standard perturbation calibration used by current GRU diagnostics.",
                "The tracked note predates native regeneration-spec pointers; this spec is the durable rerun handle.",
            ],
            repo_root=repo_root,
        )
        _write_index(
            repo_root
            / "results"
            / experiment
            / "notes"
            / f"diagnostic_regeneration_specs_{experiment}_posthoc.json",
            experiment=experiment,
            created=[repo_relative(spec_path, repo_root=repo_root)],
            repo_root=repo_root,
        )
    return [repo_relative(spec_path, repo_root=repo_root)]


def classify_manifest(name: str, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return diagnostic metadata for manifests in the active GRU lane."""

    schema = str(manifest.get("schema_version") or manifest.get("format") or "")
    if schema.startswith("rlrmp.cs_gru_standard_certificates") or name.startswith(
        "gru_standard_certificates"
    ):
        return None
    # Historical GRU evaluation-diagnostics manifests came from a retired direct
    # writer. They are parity oracles, not regeneration targets: current reruns
    # must be authored as ``rlrmp.eval.gru_diagnostics`` EvaluationRunSpecs.
    if schema.startswith("rlrmp.gru_evaluation_diagnostics") or name.startswith(
        "gru_evaluation_diagnostics"
    ):
        return None
    if schema.startswith("rlrmp.gru_postrun_materialization") or name.startswith(
        "gru_postrun_materialization"
    ):
        return {
            "diagnostic_name": "gru_postrun_materialization_bundle",
            "materializer": "rlrmp.analysis.pipelines.gru_postrun_materialization.materialize_gru_postrun_analysis",
            "source_files": [
                "src/rlrmp/analysis/pipelines/gru_postrun_materialization.py",
                "src/rlrmp/analysis/gru_standard_certificate.py",
                "src/rlrmp/eval/evaluation_diagnostics.py",
                "src/rlrmp/eval/gru_diagnostics.py",
                "src/rlrmp/eval/perturbation_bank.py",
                "src/rlrmp/eval/feedback_ablation.py",
            ],
        }
    if schema.startswith("rlrmp.objective_comparator_sidecar") or name.startswith(
        "objective_comparator"
    ):
        return {
            "diagnostic_name": "gru_objective_comparator",
            "materializer": "rlrmp.analysis.pipelines.objective_comparator.materialize_gru_objective_comparator_sidecar",
            "source_files": ["src/rlrmp/analysis/pipelines/objective_comparator.py"],
        }
    if schema.startswith("rlrmp.gru_map_error_decomposition") or name.startswith(
        "gru_map_error_decomposition"
    ):
        return {
            "diagnostic_name": "gru_map_error_decomposition",
            "materializer": "rlrmp.analysis.map_error_decomposition.map_error_decomposition_recipe",
            "source_files": ["src/rlrmp/analysis/map_error_decomposition.py"],
        }
    if schema.startswith("rlrmp.gru_perturbation_bank") or name.startswith(
        "gru_perturbation_response"
    ):
        return {
            "diagnostic_name": "gru_perturbation_response_bank",
            "materializer": (
                "rlrmp.analysis.declarative_materialization.perturbation_bank_aggregate_recipe"
            ),
            "source_files": [
                "src/rlrmp/analysis/declarative_materialization.py",
                "src/rlrmp/eval/perturbation_bank.py",
                "src/rlrmp/eval/recipes.py",
            ],
        }
    if schema.startswith("rlrmp.gru_feedback_ablation") or name.startswith("gru_feedback_ablation"):
        return {
            "diagnostic_name": "gru_feedback_ablation",
            "materializer": ("rlrmp.eval.recipes.feedback_ablation_recipe"),
            "source_files": [
                "src/rlrmp/eval/feedback_ablation.py",
                "src/rlrmp/eval/perturbation_bank.py",
            ],
        }
    if schema.startswith("rlrmp.cs_stochastic_gru.training_diagnostics_summary"):
        return {
            "diagnostic_name": "gru_training_diagnostics_summary",
            "materializer": "rlrmp.analysis.training_diagnostics.training_diagnostics_recipe",
            "source_files": ["src/rlrmp/analysis/training_diagnostics.py"],
        }
    return None


def inputs_for_manifest(
    experiment: str,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> list[dict[str, str | Path]]:
    """Infer conservative input refs from a diagnostic manifest."""

    refs: list[dict[str, str | Path]] = []
    for run_id in run_ids_for_manifest(experiment, manifest, repo_root=repo_root):
        refs.append(
            {
                "role": "run_spec",
                "path": repo_root / "results" / experiment / "runs" / run_id / "run.json",
            }
        )
        refs.append(
            {
                "role": "run_artifact_dir",
                "path": repo_root / "_artifacts" / experiment / "runs" / run_id,
            }
        )
    source_standard = manifest.get("source_standard_manifest")
    if isinstance(source_standard, str):
        refs.append({"role": "source_standard_manifest", "path": source_standard})
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        checkpoint = outputs.get("checkpoint_manifest")
        if isinstance(checkpoint, str):
            refs.append({"role": "checkpoint_manifest", "path": checkpoint})
    return refs


def outputs_for_manifest(
    experiment: str,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    classification: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, str | Path]]:
    """Infer tracked and ignored outputs for a diagnostic manifest."""

    diagnostic_name = str(classification["diagnostic_name"])
    refs: list[dict[str, str | Path]] = [
        {"role": f"{diagnostic_name}_manifest", "path": manifest_path}
    ]
    note = matching_note_path(manifest_path)
    if note.exists():
        refs.append({"role": f"{diagnostic_name}_note", "path": note})
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        for key in (
            "evaluation_bulk_dir",
            "figure_output_dir",
            "figure_summary",
        ):
            value = outputs.get(key)
            if isinstance(value, str):
                refs.append({"role": key, "path": value})
        for key in ("perturbation_response", "feedback_ablation"):
            value = outputs.get(key)
            if isinstance(value, Mapping):
                for subkey in ("bulk_dir", "json_path", "note_path"):
                    subvalue = value.get(subkey)
                    if isinstance(subvalue, str):
                        refs.append({"role": f"{key}_{subkey}", "path": subvalue})
    bulk = guessed_bulk_output(experiment, manifest_path, diagnostic_name, repo_root=repo_root)
    if bulk is not None:
        refs.append({"role": f"{diagnostic_name}_bulk", "path": bulk})
    return refs


def guessed_bulk_output(
    experiment: str,
    manifest_path: Path,
    diagnostic_name: str,
    *,
    repo_root: Path,
) -> Path | None:
    stem = manifest_path.stem.removesuffix("_manifest")
    artifact_root = repo_root / "_artifacts" / experiment
    candidates: list[Path] = []
    if diagnostic_name == "gru_perturbation_response_bank":
        tag = stem.removeprefix("gru_perturbation_response_")
        candidates.extend(
            [
                artifact_root / "perturbation_response" / f"gru_{tag}",
                artifact_root / "perturbation_response" / tag,
            ]
        )
    elif diagnostic_name == "gru_postrun_materialization_bundle":
        tag = stem.removeprefix("gru_postrun_materialization_")
        candidates.append(artifact_root / "figures" / f"gru_postrun_{tag}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_ids_for_manifest(
    experiment: str,
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[str]:
    if isinstance(manifest.get("run_ids"), list):
        values = [str(item) for item in manifest["run_ids"]]
    elif isinstance(manifest.get("runs"), Mapping):
        values = [str(item) for item in manifest["runs"].keys()]
    elif isinstance(manifest.get("source_manifests"), Mapping):
        values = [str(item) for item in manifest["source_manifests"].keys()]
    else:
        values = [
            path.parent.name
            for path in sorted((repo_root / "results" / experiment / "runs").glob("*/run.json"))
        ]
    return [item for item in values if "smoke" not in item]


def matching_note_path(manifest_path: Path) -> Path:
    stem = manifest_path.stem
    if stem.endswith("_manifest"):
        stem = stem.removesuffix("_manifest")
    return manifest_path.with_name(f"{stem}.md")


def _should_backfill_manifest(path: Path) -> bool:
    name = path.name
    if "smoke" in name:
        return False
    if "regeneration_spec" in name:
        return False
    if name.startswith("diagnostic_regeneration_specs"):
        return False
    if name.startswith("model.") or name == "run.json":
        return False
    return True


def regeneration_spec_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_regeneration_spec.json")


def _add_regeneration_pointer(path: Path, spec_path: Path, *, repo_root: Path) -> None:
    manifest = _read_json(path)
    manifest["regeneration_spec"] = repo_relative(spec_path, repo_root=repo_root)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_index(
    path: Path,
    *,
    experiment: str,
    created: Sequence[str],
    repo_root: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": BACKFILL_SCHEMA_VERSION,
                "issue": experiment,
                "tracking_issue": TRACKING_ISSUE,
                "posthoc_backfill": True,
                "created_or_updated_specs": sorted(dict.fromkeys(created)),
                "scope": "active_gru_diagnostics_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
