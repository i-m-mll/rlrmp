from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "post_run.sh"


def run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def run_command_result(
    command: list[str], *, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def init_git_repo(path: Path) -> None:
    path.mkdir()
    run_command(["git", "init"], cwd=path)
    run_command(["git", "config", "user.email", "test@example.invalid"], cwd=path)
    run_command(["git", "config", "user.name", "Post Run Test"], cwd=path)
    run_command(["git", "commit", "--allow-empty", "-m", "init"], cwd=path)


def write_json(path: Path, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return sha256(encoded.encode("utf-8")).hexdigest()


def write_run_source(path: Path, *, mismatched_spec_hash: bool = False) -> None:
    path.mkdir()
    manifest_relpath = "training_run_manifest.json"
    write_json(
        path / "graph_spec.json",
        {
            "schema_version": "feedbax.graph_spec.v1",
            "nodes": {},
            "output_ports": [],
        },
    )
    run_spec = {
        "run": "fixture__ok",
        "issue": "731fdf7",
        "schema_version": "rlrmp.run_spec.v1",
        "training_summary": {
            "training_mode": "nominal",
            "completed_batches": 3,
            "n_train_batches": 3,
            "final_validation_loss": 0.125,
        },
        "feedbax_graph": {
            "graph_spec_path": "graph_spec.json",
        },
        "provenance": {
            "rlrmp_commit": "rlrmp-fixture-sha",
            "feedbax_commit": "feedbax-fixture-sha",
            "feedbax_manifest": manifest_relpath,
            "feedbax_manifest_root": "_artifacts/feedbax_runs",
            "schema_versions": {
                "feedbax_manifest": "feedbax.manifest.v1",
                "feedbax_provider": "feedbax-provider.v1",
                "rlrmp_run_spec": "rlrmp.run_spec.v1",
            },
        },
    }
    run_spec_sha = write_json(path / "run.json", run_spec)
    write_json(path / "training_summary.json", run_spec["training_summary"])
    manifest_run_spec_sha = "bad-fixture-sha" if mismatched_spec_hash else run_spec_sha
    manifest = {
        "kind": "TrainingRunManifest",
        "schema_version": "feedbax.manifest.v1",
        "id": "feedbax-training-run:fixture-ok",
        "created_at": "2026-06-11T00:00:00Z",
        "feedbax_version": "fixture",
        "provider_version": "feedbax-provider.v1",
        "status": "completed",
        "job_id": "fixture__ok",
        "provenance": {
            "source_repo": "https://github.com/i-m-mll/rlrmp.git",
            "source_branch": "feature/64d3059-post-run-provenance",
            "source_commit": "rlrmp-fixture-sha",
            "dirty": False,
            "issues": ["731fdf7", "64d3059"],
        },
        "artifacts": [
            {
                "role": "tracked_run_spec",
                "logical_name": "fixture__ok.json",
                "sha256": manifest_run_spec_sha,
                "media_type": "application/json",
                "storage_backend": "rlrmp-results",
                "uri": "results/731fdf7/runs/fixture__ok.json",
                "metadata": {"availability": "checked_in"},
            },
            {
                "role": "training_checkpoint",
                "logical_name": "trained_model.eqx",
                "media_type": "application/x-equinox",
                "storage_backend": "rlrmp-_artifacts",
                "uri": "_artifacts/731fdf7/runs/fixture__ok/trained_model.eqx",
                "metadata": {
                    "availability": "reference_only",
                    "manifest_root": "_artifacts/feedbax_runs/",
                },
            },
        ],
        "summary_metrics": {"completed_batches": 3, "final_validation_loss": 0.125},
    }
    write_json(path / manifest_relpath, manifest)
    write_json(
        path
        / "feedbax_runs"
        / "manifests"
        / "training_runs"
        / "feedbax_training_run_fixture__ok.json",
        manifest,
    )
    (path / "trained_model.eqx").write_text("fixture model\n", encoding="utf-8")


def write_fake_commands(bin_dir: Path, auth_record: Path) -> tuple[Path, Path]:
    bin_dir.mkdir()
    fake_agent_commit = bin_dir / "agent-commit"
    fake_agent_commit.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
message=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --issue)
            shift 2
            ;;
        -m)
            message="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
git commit -m "$message"
""",
        encoding="utf-8",
    )
    fake_agent_commit.chmod(0o755)

    fake_mandible = bin_dir / "mandible"
    checkpoint_record = Path(str(auth_record) + ".checkpoint")
    fake_mandible.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
# `mandible issue checkpoint add ... --payload-file -` reads JSON from stdin;
# capture it separately. All other invocations (auth request) record their
# args to the auth-record file.
if [[ "$1" == "issue" && "$2" == "checkpoint" && "$3" == "add" ]]; then
    {{ printf 'ARGS: %s\\n' "$*"; echo '---PAYLOAD---'; cat; }} > {str(checkpoint_record)!r}
else
    printf '%s\\n' "$*" > {str(auth_record)!r}
fi
""",
        encoding="utf-8",
    )
    fake_mandible.chmod(0o755)
    return fake_agent_commit, fake_mandible


def test_dry_run_prints_actions_without_writing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    init_git_repo(repo)
    write_run_source(source)

    output = run_command(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
    )

    assert "DRY-RUN: mkdir -p" in output
    assert "DRY-RUN: rsync -a" in output
    assert "DRY-RUN: would write/verify tracked run spec" in output
    assert "rlrmp" in output.lower()
    assert "feedbax" in output.lower()
    assert "sha" in output.lower()
    assert "graphspec" in output.lower() or "graph spec" in output.lower()
    assert "feedbax.manifest.v1" in output
    assert "feedbax-provider.v1" in output
    assert "_artifacts/feedbax_runs" in output
    assert "agent-commit --issue 731fdf7" in output
    assert "mandible auth request" in output
    assert "Checklist: decide whether this run requires coordination comments" in output
    assert not (repo / "results").exists()
    assert not (repo / "_artifacts").exists()


def test_local_fixture_syncs_spec_commits_and_records_auth(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    bin_dir = tmp_path / "bin"
    auth_record = tmp_path / "auth_args.txt"
    init_git_repo(repo)
    write_run_source(source)
    fake_agent_commit, fake_mandible = write_fake_commands(bin_dir, auth_record)

    env = os.environ.copy()
    env["POST_RUN_AGENT_COMMIT"] = str(fake_agent_commit)
    env["POST_RUN_MANDIBLE_AUTH"] = str(fake_mandible)

    output = run_command(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    spec_path = repo / "results" / "731fdf7" / "runs" / "fixture__ok.json"
    artifact_dir = repo / "_artifacts" / "731fdf7" / "runs" / "fixture__ok"
    feedbax_runs_dir = repo / "_artifacts" / "feedbax_runs"
    stamped_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert stamped_spec["run"] == "fixture__ok"
    post_run_provenance = stamped_spec["post_run_provenance"]
    assert post_run_provenance["schema_version"] == "rlrmp.post_run_provenance.v1"
    assert post_run_provenance["feedbax_manifest_root"]["path"] == "_artifacts/feedbax_runs"
    assert post_run_provenance["schemas"]["feedbax_manifest"] == "feedbax.manifest.v1"
    assert post_run_provenance["schemas"]["feedbax_provider"] == "feedbax-provider.v1"
    assert (artifact_dir / "training_summary.json").is_file()
    assert (artifact_dir / "trained_model.eqx").is_file()
    assert feedbax_runs_dir.is_dir()
    assert (
        feedbax_runs_dir / "manifests" / "training_runs" / "feedbax_training_run_fixture__ok.json"
    ).is_file()
    assert "| `completed_batches` | 3 |" in output
    assert "| `final_validation_loss` | 0.125 |" in output

    log = run_command(["git", "log", "--oneline", "-1"], cwd=repo)
    assert "record post-run spec fixture__ok" in log
    committed = run_command(["git", "show", "--name-only", "--format=", "HEAD"], cwd=repo)
    assert "results/731fdf7/runs/fixture__ok.json" in committed
    assert "_artifacts/731fdf7/runs/fixture__ok" not in committed

    auth_args = auth_record.read_text(encoding="utf-8")
    assert "auth request" in auth_args
    assert "--issue 731fdf7" in auth_args
    assert "--no-watch" in auth_args

    # Exactly one terminal run-status checkpoint is emitted (W10/e8b5b3b).
    checkpoint_record = Path(str(auth_record) + ".checkpoint")
    assert checkpoint_record.is_file()
    checkpoint_text = checkpoint_record.read_text(encoding="utf-8")
    assert "issue checkpoint add 731fdf7 --kind run-status --payload-file -" in checkpoint_text
    args_line, _, payload_text = checkpoint_text.partition("---PAYLOAD---")
    assert "--kind run-status" in args_line
    payload = json.loads(payload_text)
    assert payload["kind"] == "run-status"
    assert payload["schema_version"] == 1
    assert payload["phase"] == "completed"
    assert payload["run_id"] == "fixture__ok"
    assert payload["run_spec_path"] == "results/731fdf7/runs/fixture__ok.json"
    assert payload["artifact_dir"] == "_artifacts/731fdf7/runs/fixture__ok"
    # metrics_summary references scalar summary fields, not hyperparameters.
    assert payload["metrics_summary"]["completed_batches"] == 3
    assert payload["metrics_summary"]["final_validation_loss"] == 0.125
    assert "timestamp" in payload


def test_dry_run_previews_run_status_checkpoint_without_writing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    init_git_repo(repo)
    write_run_source(source)

    output = run_command(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
    )

    assert "Run-status checkpoint (preview, not written)" in output
    assert "issue checkpoint add 731fdf7 --kind run-status --payload-file -" in output
    # The previewed payload JSON is printed (kind + schema_version visible).
    assert '"kind": "run-status"' in output
    assert '"schema_version": 1' in output
    assert '"phase": "completed"' in output
    assert '"run_spec_path": "results/731fdf7/runs/fixture__ok.json"' in output
    # Nothing is written on a dry run.
    assert not (repo / "results").exists()
    assert not (repo / "_artifacts").exists()


def test_legacy_nested_run_spec_raises_explicit_error(tmp_path: Path) -> None:
    """W8/e926665: a legacy nested results/<hash>/runs/<run>/run.json recipe is
    refused with an explicit error rather than silently promoted to the flat
    canonical path."""
    repo = tmp_path / "repo"
    bin_dir = tmp_path / "bin"
    auth_record = tmp_path / "auth_args.txt"
    init_git_repo(repo)
    fake_agent_commit, fake_mandible = write_fake_commands(bin_dir, auth_record)

    # Seed the artifact dir (with training_summary) and the LEGACY nested spec,
    # but no flat recipe and no run.json in the artifact dir.
    hash_ = "731fdf7"
    artifact_dir = repo / "_artifacts" / hash_ / "runs" / "fixture__ok"
    artifact_dir.mkdir(parents=True)
    write_json(
        artifact_dir / "training_summary.json",
        {"completed_batches": 3, "final_validation_loss": 0.125},
    )
    legacy_spec = repo / "results" / hash_ / "runs" / "fixture__ok" / "run.json"
    write_json(legacy_spec, {"run": "fixture__ok"})

    env = os.environ.copy()
    env["POST_RUN_AGENT_COMMIT"] = str(fake_agent_commit)
    env["POST_RUN_MANDIBLE_AUTH"] = str(fake_mandible)

    result = run_command_result(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            hash_,
            "--run",
            "fixture__ok",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode != 0
    assert "legacy nested run spec" in result.stdout
    assert "fixture__ok/run.json" in result.stdout
    assert not auth_record.exists()
    assert not Path(str(auth_record) + ".checkpoint").exists()


def test_rejects_unpinned_feedbax_runs_dir_before_writing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    init_git_repo(repo)
    write_run_source(source)

    env = os.environ.copy()
    env["FEEDBAX_RUNS_DIR"] = str(tmp_path / "wrong_feedbax_runs")

    result = run_command_result(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode != 0
    assert "FEEDBAX_RUNS_DIR must be pinned" in result.stdout
    assert "_artifacts/feedbax_runs" in result.stdout
    assert not (repo / "results").exists()
    assert not (repo / "_artifacts").exists()


def test_manifest_run_spec_mismatch_fails_before_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    bin_dir = tmp_path / "bin"
    auth_record = tmp_path / "auth_args.txt"
    init_git_repo(repo)
    write_run_source(source, mismatched_spec_hash=True)
    fake_agent_commit, fake_mandible = write_fake_commands(bin_dir, auth_record)

    env = os.environ.copy()
    env["POST_RUN_AGENT_COMMIT"] = str(fake_agent_commit)
    env["POST_RUN_MANDIBLE_AUTH"] = str(fake_mandible)

    result = run_command_result(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode != 0
    assert "parity" in result.stdout.lower()
    assert "bad-fixture-sha" in result.stdout
    assert not auth_record.exists()
    log = run_command(["git", "log", "--oneline", "--all"], cwd=repo)
    assert "record post-run spec fixture__ok" not in log


def test_dirty_uv_lock_fails_before_commit_and_auth(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    bin_dir = tmp_path / "bin"
    auth_record = tmp_path / "auth_args.txt"
    init_git_repo(repo)
    (repo / "uv.lock").write_text("locked\n", encoding="utf-8")
    run_command(["git", "add", "uv.lock"], cwd=repo)
    run_command(["git", "commit", "-m", "track uv lock"], cwd=repo)
    (repo / "uv.lock").write_text("dirty\n", encoding="utf-8")
    write_run_source(source)
    fake_agent_commit, fake_mandible = write_fake_commands(bin_dir, auth_record)

    env = os.environ.copy()
    env["POST_RUN_AGENT_COMMIT"] = str(fake_agent_commit)
    env["POST_RUN_MANDIBLE_AUTH"] = str(fake_mandible)

    result = run_command_result(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode != 0
    assert "uv.lock has unstaged changes" in result.stdout
    assert "POST_RUN_ALLOW_DIRTY_UV_LOCK=1" in result.stdout
    assert not auth_record.exists()
    log = run_command(["git", "log", "--oneline", "-1"], cwd=repo)
    assert "track uv lock" in log


def test_dirty_uv_lock_emergency_override_allows_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    bin_dir = tmp_path / "bin"
    auth_record = tmp_path / "auth_args.txt"
    init_git_repo(repo)
    (repo / "uv.lock").write_text("locked\n", encoding="utf-8")
    run_command(["git", "add", "uv.lock"], cwd=repo)
    run_command(["git", "commit", "-m", "track uv lock"], cwd=repo)
    (repo / "uv.lock").write_text("dirty\n", encoding="utf-8")
    write_run_source(source)
    fake_agent_commit, fake_mandible = write_fake_commands(bin_dir, auth_record)

    env = os.environ.copy()
    env["POST_RUN_AGENT_COMMIT"] = str(fake_agent_commit)
    env["POST_RUN_MANDIBLE_AUTH"] = str(fake_mandible)
    env["POST_RUN_ALLOW_DIRTY_UV_LOCK"] = "1"

    output = run_command(
        [
            "bash",
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--issue",
            "731fdf7",
            "--run",
            "fixture__ok",
            "--artifacts-src",
            f"local:{source}",
        ],
        cwd=REPO_ROOT,
        env=env,
    )

    assert "skipping uv.lock cleanliness guard" in output
    assert auth_record.exists()
    log = run_command(["git", "log", "--oneline", "-1"], cwd=repo)
    assert "record post-run spec fixture__ok" in log
