"""Map completed orchestrated rows into the established post-run layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def map_registered_run_set(
    run_set_dir: Path,
    *,
    repo_root: Path,
    issue: str,
    run_prefix: str,
) -> tuple[Path, ...]:
    """Idempotently materialize REGISTERed rows for ``post_run.sh``."""
    registration_path = run_set_dir / "registration.json"
    if not registration_path.is_file():
        raise ValueError("run set has no registration.json")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("status") != "completed":
        raise ValueError("orchestrated post-run mapping requires completed registration")
    bundle = json.loads((run_set_dir / "bundle.json").read_text(encoding="utf-8"))
    run_set_id = str(registration["run_set_id"])
    outputs: list[Path] = []
    for row in bundle["rows"]:
        row_id = str(row["row_id"])
        source = run_set_dir / "collected" / row_id
        target = repo_root / "_artifacts" / issue / "runs" / f"{run_prefix}__{row_id}"
        target.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for name in ("manifest.json", "training-diagnostics.json", "training_summary.json"):
            source_path = source / name
            if not source_path.is_file():
                raise ValueError(f"collected row {row_id!r} is missing {name}")
            target_path = target / name
            if not target_path.exists() or target_path.read_bytes() != source_path.read_bytes():
                shutil.copy2(source_path, target_path)
            copied[name] = str(target_path.relative_to(repo_root))
        recipe = {
            "schema_id": "rlrmp.spec.orchestrated_post_run",
            "schema_version": "rlrmp.spec.orchestrated_post_run.v1",
            "issue": issue,
            "run_set_id": run_set_id,
            "row_id": row_id,
            "certificate_sha256": registration["certificate_sha256"],
            "execution_hash": row["execution"]["execution_capsule"]["execution_hash"],
            "source_paths": copied,
        }
        recipe_path = target / "run.json"
        encoded = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        recipe_path.write_text(encoded, encoding="utf-8")
        outputs.append(recipe_path)
    return tuple(outputs)


__all__ = ["map_registered_run_set"]
