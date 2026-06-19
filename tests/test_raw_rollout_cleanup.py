from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rlrmp.analysis.rollout_cleanup import (
    CleanupPreconditionError,
    cleanup_raw_perturbation_rollouts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "cleanup_perturbation_response_rollouts.py"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(repo_root: Path) -> dict[str, Path]:
    issue = "abc1234"
    notes_dir = repo_root / "results" / issue / "notes"
    bulk_dir = repo_root / "_artifacts" / issue / "perturbation_response" / "bank_a"
    run_dir = bulk_dir / "run_a"
    run_dir.mkdir(parents=True)
    raw_a = run_dir / "row_a.npz"
    raw_b = run_dir / "row_b.npz"
    raw_a.write_bytes(b"raw-a")
    raw_b.write_bytes(b"raw-bb")
    (bulk_dir / "detail_manifest.json").write_text("{}", encoding="utf-8")

    manifest_path = notes_dir / "gru_perturbation_response_bank_a_manifest.json"
    note_path = notes_dir / "gru_perturbation_response_bank_a.md"
    regeneration_path = notes_dir / "gru_perturbation_response_bank_a_manifest_regeneration_spec.json"
    note_path.parent.mkdir(parents=True)
    note_path.write_text("# Perturbation response\n", encoding="utf-8")
    _write_json(regeneration_path, {"schema_version": "rlrmp.regeneration_spec.v1"})
    _write_json(
        manifest_path,
        {
            "schema_version": "rlrmp.gru_perturbation_response.v2",
            "issue": issue,
            "regeneration_spec": f"results/{issue}/notes/{regeneration_path.name}",
            "bulk_detail_manifest": {
                "path": f"_artifacts/{issue}/perturbation_response/bank_a/detail_manifest.json",
            },
            "runs": {
                "run_a": {
                    "bulk_files": {
                        "row_a": f"_artifacts/{issue}/perturbation_response/bank_a/run_a/row_a.npz",
                        "row_b": f"_artifacts/{issue}/perturbation_response/bank_a/run_a/row_b.npz",
                    }
                }
            },
        },
    )
    return {
        "manifest": manifest_path,
        "note": note_path,
        "regeneration": regeneration_path,
        "bulk_dir": bulk_dir,
        "raw_a": raw_a,
        "raw_b": raw_b,
    }


def test_dry_run_records_candidates_without_deleting(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    manifest = cleanup_raw_perturbation_rollouts(
        tracked_manifest_path=paths["manifest"],
        repo_root=tmp_path,
        apply=False,
    )

    assert manifest["mode"] == "dry_run"
    assert manifest["candidate_file_count"] == 2
    assert manifest["candidate_bytes"] == len(b"raw-a") + len(b"raw-bb")
    assert manifest["deleted_file_count"] == 0
    assert paths["raw_a"].exists()
    assert paths["raw_b"].exists()
    precondition_roles = {item["role"] for item in manifest["kept_preconditions"]}
    assert precondition_roles == {
        "tracked_manifest",
        "tracked_summary",
        "regeneration_spec",
        "bulk_dir",
    }


def test_apply_deletes_npz_and_writes_durable_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    manifest = cleanup_raw_perturbation_rollouts(
        tracked_manifest_path=paths["manifest"],
        repo_root=tmp_path,
        apply=True,
    )

    assert manifest["mode"] == "apply"
    assert manifest["deleted_file_count"] == 2
    assert manifest["deleted_bytes"] == len(b"raw-a") + len(b"raw-bb")
    assert not paths["raw_a"].exists()
    assert not paths["raw_b"].exists()
    assert (paths["bulk_dir"] / "detail_manifest.json").exists()

    cleanup_manifest = (
        tmp_path / "results" / "abc1234" / "notes" / "raw_rollout_cleanup_bank_a.json"
    )
    payload = json.loads(cleanup_manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rlrmp.raw_rollout_cleanup.v1"
    assert payload["deleted_file_count"] == 2
    assert payload["missing_manifest_references"] == []
    assert payload["kept_preconditions"][0]["path"].startswith("results/abc1234/notes/")


def test_missing_regeneration_spec_blocks_cleanup_before_deletion(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["regeneration"].unlink()

    with pytest.raises(CleanupPreconditionError, match="preconditions"):
        cleanup_raw_perturbation_rollouts(
            tracked_manifest_path=paths["manifest"],
            repo_root=tmp_path,
            apply=True,
        )

    assert paths["raw_a"].exists()
    assert paths["raw_b"].exists()


def test_cli_defaults_to_dry_run(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(SCRIPT),
            "--repo-root",
            str(tmp_path),
            "--manifest",
            str(paths["manifest"]),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert "dry_run: would delete 2 raw rollout file(s)" in result.stdout
    assert paths["raw_a"].exists()
    assert paths["raw_b"].exists()
