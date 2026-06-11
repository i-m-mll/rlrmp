from __future__ import annotations

import json
import os
import subprocess
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


def init_git_repo(path: Path) -> None:
    path.mkdir()
    run_command(["git", "init"], cwd=path)
    run_command(["git", "config", "user.email", "test@example.invalid"], cwd=path)
    run_command(["git", "config", "user.name", "Post Run Test"], cwd=path)
    run_command(["git", "commit", "--allow-empty", "-m", "init"], cwd=path)


def write_run_source(path: Path) -> None:
    path.mkdir()
    (path / "run.json").write_text(
        json.dumps(
            {
                "run": "fixture__ok",
                "training_summary": {"training_mode": "nominal"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "training_summary.json").write_text(
        json.dumps(
            {
                "training_mode": "nominal",
                "completed_batches": 3,
                "n_train_batches": 3,
                "final_validation_loss": 0.125,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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
    fake_mandible.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > {str(auth_record)!r}
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
    assert json.loads(spec_path.read_text(encoding="utf-8"))["run"] == "fixture__ok"
    assert (artifact_dir / "training_summary.json").is_file()
    assert (artifact_dir / "trained_model.eqx").is_file()
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
