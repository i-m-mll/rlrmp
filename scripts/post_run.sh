#!/usr/bin/env bash
# Deterministic post-training-run protocol wrapper.
#
# Usage:
#   scripts/post_run.sh --issue <id> --run <label> [--artifacts-src <source>] [--dry-run]
#
# Source forms:
#   local:/path/to/run-dir      Copy a local run directory into _artifacts/.
#   /path/to/run-dir            Same as local:/path/to/run-dir.
#   modal[:volume-name]         Pull results/ and _artifacts/ paths from a Modal volume.
#   pod:<rsync-source>          Rsync from an SSH source such as host:/path/to/run-dir/.
#
# Emergency override:
#   POST_RUN_ALLOW_DIRTY_UV_LOCK=1 skips the uv.lock cleanliness guard. Use only
#   when the lockfile change is deliberate and already accounted for elsewhere.

set -euo pipefail

usage() {
    sed -n '2,13p' "$0" >&2
}

die() {
    echo "error: $*" >&2
    exit 1
}

quote_command() {
    local quoted=()
    local arg
    for arg in "$@"; do
        printf -v arg "%q" "$arg"
        quoted+=("$arg")
    done
    printf '%s\n' "${quoted[*]}"
}

run_or_print() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN: $(quote_command "$@")"
        return 0
    fi
    echo "+ $(quote_command "$@")"
    "$@"
}

run_in_repo_or_print() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY-RUN: cd $(printf '%q' "$REPO_ROOT") && $(quote_command "$@")"
        return 0
    fi
    echo "+ cd $(printf '%q' "$REPO_ROOT") && $(quote_command "$@")"
    (cd "$REPO_ROOT" && "$@")
}

require_clean_index() {
    if ! git -C "$REPO_ROOT" diff --cached --quiet; then
        die "git index is not clean; commit or unstage existing changes before post_run.sh"
    fi
}

require_uv_lock_clean() {
    local uv_lock="$REPO_ROOT/uv.lock"
    [[ -e "$uv_lock" ]] || return 0
    if [[ "${POST_RUN_ALLOW_DIRTY_UV_LOCK:-0}" == "1" ]]; then
        echo "warning: POST_RUN_ALLOW_DIRTY_UV_LOCK=1; skipping uv.lock cleanliness guard" >&2
        return 0
    fi
    if ! git -C "$REPO_ROOT" ls-files --error-unmatch uv.lock >/dev/null 2>&1; then
        die "uv.lock exists but is not tracked; refusing post-run commit/auth work"
    fi
    if ! git -C "$REPO_ROOT" diff --quiet -- uv.lock; then
        die "uv.lock has unstaged changes; set POST_RUN_ALLOW_DIRTY_UV_LOCK=1 only for an emergency"
    fi
    if ! git -C "$REPO_ROOT" diff --cached --quiet -- uv.lock; then
        die "uv.lock has staged changes; set POST_RUN_ALLOW_DIRTY_UV_LOCK=1 only for an emergency"
    fi
}

validate_id_component() {
    local name="$1"
    local value="$2"
    [[ -n "$value" ]] || die "$name must not be empty"
    [[ "$value" != *"/"* ]] || die "$name must not contain /: $value"
    [[ "$value" != *".."* ]] || die "$name must not contain ..: $value"
}

render_metrics_table() {
    local summary_path="$1"
    [[ -f "$summary_path" ]] || die "missing training summary: $summary_path"
    (
        cd "$SCRIPT_REPO_ROOT"
        uv run --no-sync python - "$summary_path" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
priority_keys = [
    "training_mode",
    "loss_objective",
    "completed_batches",
    "n_train_batches",
    "best_batch",
    "final_train_loss",
    "final_validation_loss",
    "best_validation_loss",
    "validation_loss",
    "training_duration_seconds",
    "training_batches_per_second",
]

rows: list[tuple[str, Any]] = []
seen: set[str] = set()
for key in priority_keys:
    if key in summary and scalar(summary[key]):
        rows.append((key, summary[key]))
        seen.add(key)

for key in sorted(summary):
    if key in seen or not scalar(summary[key]):
        continue
    if len(rows) >= 14:
        break
    rows.append((key, summary[key]))

print("| Metric | Value |")
print("|---|---|")
if not rows:
    print("| summary | present, no scalar metrics found |")
else:
    for key, value in rows:
        print(f"| `{key}` | {format_value(value)} |")
PY
    )
}

# Build the run-status checkpoint payload JSON (kind=run-status,
# schema_version=1) on stdout. The payload references the tracked run.json by
# repo-relative path and embeds only the scalar metrics summary — never the
# run hyperparameters. See CLAUDE.md RunPod runbook §9 (run-status checkpoint
# convention) and issue e8b5b3b.
build_run_status_payload() {
    local phase="$1"
    local summary_path="$2"
    (
        cd "$SCRIPT_REPO_ROOT"
        uv run --no-sync python - \
            "$phase" "$RUN_LABEL" "$REPO_ROOT" "$SPEC_PATH" "$ARTIFACT_DIR" "$summary_path" <<'PY'
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path
from typing import Any

phase = sys.argv[1]
run_id = sys.argv[2]
repo_root = Path(sys.argv[3]).resolve()
spec_path = Path(sys.argv[4])
artifact_dir = Path(sys.argv[5])
summary_path = Path(sys.argv[6])


def rel(path: Path) -> str:
    abs_path = path if path.is_absolute() else (repo_root / path)
    try:
        return str(abs_path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


metrics_summary: dict[str, Any] = {}
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    priority_keys = [
        "training_mode",
        "loss_objective",
        "completed_batches",
        "n_train_batches",
        "best_batch",
        "final_train_loss",
        "final_validation_loss",
        "best_validation_loss",
        "validation_loss",
        "training_duration_seconds",
        "training_batches_per_second",
    ]
    for key in priority_keys:
        if key in summary and scalar(summary[key]):
            metrics_summary[key] = clean(summary[key])

payload = {
    "kind": "run-status",
    "schema_version": 1,
    "run_id": run_id,
    "phase": phase,
    "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "artifact_dir": rel(artifact_dir),
    "run_spec_path": rel(spec_path),
    "metrics_summary": metrics_summary,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    )
}

run_post_run_contract() {
    local mode="$1"
    local input_spec_path="$2"
    (
        cd "$SCRIPT_REPO_ROOT"
        uv run --no-sync python - "$mode" "$REPO_ROOT" "$SPEC_PATH" "$input_spec_path" \
            "$ARTIFACT_DIR" "$ISSUE" "$RUN_LABEL" "$FEEDBAX_MANIFEST_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from feedbax.contracts.manifest import (
    PROVIDER_VERSION,
    SCHEMA_VERSION,
    TrainingRunManifest,
    load_manifest,
)


POST_RUN_SCHEMA_VERSION = "rlrmp.post_run_provenance.v1"
PINNED_MANIFEST_ROOT = "_artifacts/feedbax_runs"


def die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path, root: Path) -> str:
    root_absolute = root.absolute()
    path_absolute = path if path.is_absolute() else root_absolute / path
    try:
        return str(path_absolute.relative_to(root_absolute))
    except ValueError:
        return str(path.resolve().relative_to(root.resolve()))


def git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def git_record(repo: Path) -> dict[str, Any]:
    status = git_value(repo, "status", "--short")
    return {
        "commit": git_value(repo, "rev-parse", "HEAD"),
        "branch": git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "remote": git_value(repo, "config", "--get", "remote.origin.url"),
    }


def feedbax_repo(script_repo_root: Path) -> Path | None:
    pyproject = load_json_compatible_toml(script_repo_root / "pyproject.toml")
    source = (
        pyproject.get("tool", {})
        .get("uv", {})
        .get("sources", {})
        .get("feedbax", {})
        .get("path")
    )
    if not source:
        return None
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = script_repo_root / path
    return path if path.exists() else None


def load_json_compatible_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def graph_metadata(run_spec: dict[str, Any], spec_path: Path, artifact_dir: Path) -> dict[str, Any]:
    graph = run_spec.get("feedbax_graph")
    if not isinstance(graph, dict):
        return {
            "graph_spec_path": None,
            "graph_spec_sha256": None,
            "graph_spec_version": None,
            "graph_manifest_path": None,
            "graph_manifest_sha256": None,
        }

    def find_sidecar(value: Any) -> Path | None:
        if value in (None, ""):
            return None
        raw = Path(str(value))
        candidates = [raw] if raw.is_absolute() else [
            spec_path.parent / raw,
            artifact_dir / raw,
            artifact_dir / raw.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    graph_path = find_sidecar(graph.get("graph_spec_path"))
    manifest_path = find_sidecar(graph.get("manifest_path"))
    graph_version = None
    if graph_path is not None:
        try:
            graph_payload = load_json(graph_path)
            graph_version = (
                graph_payload.get("schema_version")
                or graph_payload.get("version")
                or graph_payload.get("$schema")
            )
        except Exception:
            graph_version = None

    return {
        "graph_spec_path": graph.get("graph_spec_path"),
        "graph_spec_sha256": sha256_file(graph_path) if graph_path is not None else None,
        "graph_spec_version": graph_version,
        "graph_manifest_path": graph.get("manifest_path"),
        "graph_manifest_sha256": sha256_file(manifest_path) if manifest_path is not None else None,
    }


def provenance_stamp(
    *,
    repo_root: Path,
    script_repo_root: Path,
    spec_path: Path,
    artifact_dir: Path,
    manifest_root: Path,
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    feedbax_path = feedbax_repo(script_repo_root)
    feedbax_git = git_record(feedbax_path) if feedbax_path is not None else {
        "commit": None,
        "branch": None,
        "dirty": None,
        "remote": None,
    }
    return {
        "schema_version": POST_RUN_SCHEMA_VERSION,
        "tool": "scripts/post_run.sh",
        "rlrmp": git_record(repo_root),
        "feedbax": feedbax_git,
        "schemas": {
            "post_run_provenance": POST_RUN_SCHEMA_VERSION,
            "feedbax_manifest": SCHEMA_VERSION,
            "feedbax_provider": PROVIDER_VERSION,
        },
        "feedbax_manifest_root": {
            "path": PINNED_MANIFEST_ROOT,
            "absolute_path_sha256": hashlib.sha256(str(manifest_root.resolve()).encode()).hexdigest(),
            "env": "FEEDBAX_RUNS_DIR",
        },
        "feedbax_graph": graph_metadata(run_spec, spec_path, artifact_dir),
    }


def iter_training_manifests(manifest_root: Path, artifact_dir: Path) -> list[Path]:
    paths: list[Path] = []
    root_dir = manifest_root / "manifests" / "training_runs"
    if root_dir.exists():
        paths.extend(sorted(root_dir.glob("*.json")))
    for name in (
        "training_run_manifest.json",
        "feedbax_training_run_manifest.json",
        "model.training_run.manifest.json",
    ):
        candidate = artifact_dir / name
        if candidate.is_file():
            paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def manifest_refs(manifest: TrainingRunManifest) -> set[str]:
    refs: set[str] = set()
    if manifest.training_spec is not None and manifest.training_spec.ref:
        refs.add(manifest.training_spec.ref)
    for artifact in manifest.artifacts:
        if artifact.uri:
            refs.add(artifact.uri)
        original_uri = artifact.metadata.get("original_uri")
        if isinstance(original_uri, str):
            refs.add(original_uri)
    return refs


def manifest_matches(
    manifest: TrainingRunManifest,
    *,
    rel_spec_path: str,
    run_label: str,
) -> bool:
    if rel_spec_path in manifest_refs(manifest):
        return True
    if manifest.job_id == run_label:
        return True
    return manifest.id.endswith(run_label)


def comparable_training_spec(run_spec: dict[str, Any]) -> dict[str, Any]:
    comparable = {
        key: run_spec.get(key)
        for key in ("run", "issue", "training_summary", "loss_objective")
        if key in run_spec
    }
    if not comparable and "training_summary" in run_spec:
        comparable["training_summary"] = run_spec["training_summary"]
    return comparable


def check_inline_manifest_parity(manifest: TrainingRunManifest, run_spec: dict[str, Any]) -> None:
    inline = manifest.training_spec.inline if manifest.training_spec is not None else None
    if not inline:
        return
    comparable = comparable_training_spec(run_spec)
    for key, value in comparable.items():
        if key in inline and inline[key] != value:
            die(
                "Feedbax TrainingRunManifest parity failed: "
                f"training_spec.inline.{key}={inline[key]!r} does not match run spec {value!r}"
            )


def update_manifest_hashes(
    manifest_path: Path,
    *,
    actual_sha: str,
    actual_size: int,
    rel_spec_path: str,
) -> None:
    payload = load_json(manifest_path)
    training_spec = payload.get("training_spec")
    if isinstance(training_spec, dict) and training_spec.get("ref") == rel_spec_path:
        training_spec["sha256"] = actual_sha
    for artifact in payload.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        if artifact.get("role") == "tracked_run_spec" and artifact.get("uri") == rel_spec_path:
            artifact["sha256"] = actual_sha
            artifact["size_bytes"] = actual_size
    write_json(manifest_path, payload)


def parity_check(
    *,
    repo_root: Path,
    spec_path: Path,
    artifact_dir: Path,
    manifest_root: Path,
    issue: str,
    run_label: str,
    update_hashes: bool,
) -> str:
    rel_spec_path = rel(spec_path, repo_root)
    manifest_paths = iter_training_manifests(manifest_root, artifact_dir)
    matches_by_id: dict[str, tuple[Path, TrainingRunManifest]] = {}
    for path in manifest_paths:
        manifest = load_manifest(path)
        if not isinstance(manifest, TrainingRunManifest):
            continue
        if manifest_matches(manifest, rel_spec_path=rel_spec_path, run_label=run_label):
            previous = matches_by_id.get(manifest.id)
            if previous is None or str(path).startswith(str(manifest_root)):
                matches_by_id[manifest.id] = (path, manifest)

    matches = list(matches_by_id.values())
    if not matches:
        return "not_found"
    if len(matches) > 1:
        die(
            "Feedbax TrainingRunManifest parity failed: multiple manifests match "
            f"{rel_spec_path}: {', '.join(str(path) for path, _ in matches)}"
        )

    manifest_path, manifest = matches[0]
    if issue not in manifest.provenance.issues:
        print(
            "warning: matching Feedbax TrainingRunManifest does not list "
            f"issue {issue} in provenance.issues",
            file=sys.stderr,
        )
    check_inline_manifest_parity(manifest, load_json(spec_path))

    actual_sha = sha256_file(spec_path)
    expected_hashes: list[tuple[str, str | None]] = []
    if manifest.training_spec is not None and manifest.training_spec.ref == rel_spec_path:
        expected_hashes.append(("training_spec.sha256", manifest.training_spec.sha256))
    for artifact in manifest.artifacts:
        if artifact.role == "tracked_run_spec" and artifact.uri == rel_spec_path:
            expected_hashes.append(("tracked_run_spec.sha256", artifact.sha256))

    for label, expected in expected_hashes:
        if expected is not None and expected != actual_sha and not update_hashes:
            die(
                "Feedbax TrainingRunManifest parity failed: "
                f"{label}={expected} does not match {rel_spec_path} sha256={actual_sha}"
            )

    if update_hashes:
        update_manifest_hashes(
            manifest_path,
            actual_sha=actual_sha,
            actual_size=spec_path.stat().st_size,
            rel_spec_path=rel_spec_path,
        )
    return str(manifest_path)


def print_stamp_summary(stamp: dict[str, Any], parity: str | None) -> None:
    graph = stamp["feedbax_graph"]
    print("Provenance stamps")
    print(f"  rlrmp SHA: {stamp['rlrmp']['commit'] or 'unavailable'}")
    print(f"  feedbax SHA: {stamp['feedbax']['commit'] or 'unavailable'}")
    print(f"  GraphSpec hash: {graph['graph_spec_sha256'] or 'unavailable'}")
    print(f"  GraphSpec version: {graph['graph_spec_version'] or 'unavailable'}")
    print(f"  Feedbax manifest schema: {stamp['schemas']['feedbax_manifest']}")
    print(f"  Feedbax provider schema: {stamp['schemas']['feedbax_provider']}")
    print(f"  post_run schema: {stamp['schemas']['post_run_provenance']}")
    print(f"  pinned Feedbax manifest root: {stamp['feedbax_manifest_root']['path']}")
    if parity is not None:
        print(f"  TrainingRunManifest parity: {parity}")


def main() -> None:
    mode = sys.argv[1]
    repo_root = Path(sys.argv[2]).resolve()
    spec_path = Path(sys.argv[3]).resolve()
    input_spec = Path(sys.argv[4]).resolve() if sys.argv[4] else spec_path
    artifact_dir = Path(sys.argv[5]).resolve()
    issue = sys.argv[6]
    run_label = sys.argv[7]
    manifest_root = Path(sys.argv[8])
    script_repo_root = Path.cwd().resolve()

    if rel(manifest_root, repo_root) != PINNED_MANIFEST_ROOT:
        die(f"FEEDBAX_RUNS_DIR must resolve to {PINNED_MANIFEST_ROOT}; found {manifest_root}")

    source_spec = spec_path if spec_path.is_file() else input_spec
    if not source_spec.is_file():
        if mode == "dry-run":
            run_spec: dict[str, Any] = {}
        else:
            die(f"missing run spec for post-run provenance: {source_spec}")
    else:
        run_spec = load_json(source_spec)

    graph_artifact_dir = source_spec.parent if mode == "dry-run" and source_spec.is_file() else artifact_dir
    stamp = provenance_stamp(
        repo_root=repo_root,
        script_repo_root=script_repo_root,
        spec_path=spec_path,
        artifact_dir=graph_artifact_dir,
        manifest_root=manifest_root,
        run_spec=run_spec,
    )

    parity = None
    if mode == "dry-run":
        parity = "would check pinned root before commit"
        print_stamp_summary(stamp, parity)
        return

    parity_before_stamp = parity_check(
        repo_root=repo_root,
        spec_path=spec_path,
        artifact_dir=artifact_dir,
        manifest_root=manifest_root,
        issue=issue,
        run_label=run_label,
        update_hashes=False,
    )
    run_spec["post_run_provenance"] = stamp
    write_json(spec_path, run_spec)
    parity_after_stamp = parity_check(
        repo_root=repo_root,
        spec_path=spec_path,
        artifact_dir=artifact_dir,
        manifest_root=manifest_root,
        issue=issue,
        run_label=run_label,
        update_hashes=True,
    )
    parity = parity_after_stamp if parity_before_stamp == parity_after_stamp else (
        f"pre-stamp {parity_before_stamp}; stamped {parity_after_stamp}"
    )
    print_stamp_summary(stamp, parity)


if __name__ == "__main__":
    main()
PY
    )
}

ISSUE=""
RUN_LABEL=""
ARTIFACTS_SRC=""
DRY_RUN=0
REPO_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --issue)
            [[ $# -ge 2 ]] || die "--issue requires a value"
            ISSUE="$2"
            shift 2
            ;;
        --run)
            [[ $# -ge 2 ]] || die "--run requires a value"
            RUN_LABEL="$2"
            shift 2
            ;;
        --artifacts-src)
            [[ $# -ge 2 ]] || die "--artifacts-src requires a value"
            ARTIFACTS_SRC="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --repo-root)
            [[ $# -ge 2 ]] || die "--repo-root requires a value"
            REPO_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$ISSUE" ]] || die "--issue is required"
[[ -n "$RUN_LABEL" ]] || die "--run is required"
validate_id_component "issue" "$ISSUE"
validate_id_component "run" "$RUN_LABEL"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "$REPO_ROOT" ]]; then
    REPO_ROOT="$SCRIPT_REPO_ROOT"
else
    REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
fi

HASH="${ISSUE:0:7}"
RUNS_DIR="$REPO_ROOT/results/$HASH/runs"
SPEC_PATH="$RUNS_DIR/$RUN_LABEL.json"
LEGACY_SPEC_DIR="$RUNS_DIR/$RUN_LABEL"
ARTIFACT_DIR="$REPO_ROOT/_artifacts/$HASH/runs/$RUN_LABEL"
SUMMARY_PATH="$ARTIFACT_DIR/training_summary.json"

AGENT_COMMIT="${POST_RUN_AGENT_COMMIT:-agent-commit}"
MANDIBLE="${POST_RUN_MANDIBLE_AUTH:-mandible}"
MODAL_VOLUME_DEFAULT="${POST_RUN_MODAL_VOLUME:-rlrmp-cs-stochastic-gru}"
PINNED_FEEDBAX_MANIFEST_ROOT="$REPO_ROOT/_artifacts/feedbax_runs"
if [[ -n "${FEEDBAX_RUNS_DIR:-}" ]]; then
    CONFIGURED_FEEDBAX_MANIFEST_ROOT="$(cd "$(dirname "$FEEDBAX_RUNS_DIR")" && pwd)/$(basename "$FEEDBAX_RUNS_DIR")"
    [[ "$CONFIGURED_FEEDBAX_MANIFEST_ROOT" == "$PINNED_FEEDBAX_MANIFEST_ROOT" ]] || {
        die "FEEDBAX_RUNS_DIR must be pinned to $PINNED_FEEDBAX_MANIFEST_ROOT; found $FEEDBAX_RUNS_DIR"
    }
fi
export FEEDBAX_RUNS_DIR="$PINNED_FEEDBAX_MANIFEST_ROOT"
FEEDBAX_MANIFEST_ROOT="$FEEDBAX_RUNS_DIR"

echo "Post-run protocol"
echo "  issue: $ISSUE"
echo "  run: $RUN_LABEL"
echo "  run spec: $SPEC_PATH"
echo "  artifacts: $ARTIFACT_DIR"
echo "  feedbax manifests: $FEEDBAX_MANIFEST_ROOT"

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_clean_index
fi

case "$ARTIFACTS_SRC" in
    "")
        echo "Sync: using existing local artifact/spec paths."
        INPUT_SPEC_PATH=""
        SOURCE_FEEDBAX_RUNS_DIR=""
        ;;
    local:*)
        SOURCE_DIR="${ARTIFACTS_SRC#local:}"
        [[ -d "$SOURCE_DIR" ]] || die "local artifact source does not exist: $SOURCE_DIR"
        INPUT_SPEC_PATH="$SOURCE_DIR/run.json"
        SOURCE_FEEDBAX_RUNS_DIR="$SOURCE_DIR/feedbax_runs"
        run_or_print mkdir -p "$ARTIFACT_DIR"
        run_or_print rsync -a --exclude feedbax_runs/ "$SOURCE_DIR"/ "$ARTIFACT_DIR"/
        ;;
    modal|modal:*)
        VOLUME_NAME="$MODAL_VOLUME_DEFAULT"
        if [[ "$ARTIFACTS_SRC" == modal:* && "$ARTIFACTS_SRC" != "modal:" ]]; then
            VOLUME_NAME="${ARTIFACTS_SRC#modal:}"
        fi
        INPUT_SPEC_PATH=""
        SOURCE_FEEDBAX_RUNS_DIR=""
        run_or_print mkdir -p "$RUNS_DIR" "$ARTIFACT_DIR"
        run_or_print modal volume get --force "$VOLUME_NAME" \
            "results/$HASH/runs/$RUN_LABEL" "$RUNS_DIR"
        run_or_print modal volume get --force "$VOLUME_NAME" \
            "_artifacts/$HASH/runs/$RUN_LABEL" "$ARTIFACT_DIR/.."
        ;;
    pod:*)
        RSYNC_SOURCE="${ARTIFACTS_SRC#pod:}"
        [[ -n "$RSYNC_SOURCE" ]] || die "pod source must include an rsync source"
        INPUT_SPEC_PATH=""
        SOURCE_FEEDBAX_RUNS_DIR=""
        run_or_print mkdir -p "$ARTIFACT_DIR"
        run_or_print rsync -az --stats --no-owner --no-group \
            "$RSYNC_SOURCE"/ "$ARTIFACT_DIR"/
        ;;
    *)
        if [[ -d "$ARTIFACTS_SRC" ]]; then
            INPUT_SPEC_PATH="$ARTIFACTS_SRC/run.json"
            SOURCE_FEEDBAX_RUNS_DIR="$ARTIFACTS_SRC/feedbax_runs"
            run_or_print mkdir -p "$ARTIFACT_DIR"
            run_or_print rsync -a --exclude feedbax_runs/ "$ARTIFACTS_SRC"/ "$ARTIFACT_DIR"/
        else
            die "unknown --artifacts-src form or missing directory: $ARTIFACTS_SRC"
        fi
        ;;
esac

if [[ -n "$SOURCE_FEEDBAX_RUNS_DIR" && -d "$SOURCE_FEEDBAX_RUNS_DIR" ]]; then
    run_or_print mkdir -p "$FEEDBAX_MANIFEST_ROOT"
    run_or_print rsync -a "$SOURCE_FEEDBAX_RUNS_DIR"/ "$FEEDBAX_MANIFEST_ROOT"/
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would write/verify tracked run spec at $SPEC_PATH"
    run_post_run_contract "dry-run" "$INPUT_SPEC_PATH"
    echo "DRY-RUN: would render metrics table from $SUMMARY_PATH"
    echo "DRY-RUN: would git add $SPEC_PATH before agent-commit"
else
    mkdir -p "$FEEDBAX_MANIFEST_ROOT"
    mkdir -p "$RUNS_DIR"
    if [[ -f "$ARTIFACT_DIR/run.json" ]]; then
        cp "$ARTIFACT_DIR/run.json" "$SPEC_PATH"
    elif [[ -f "$SPEC_PATH" ]]; then
        :
    elif [[ -f "$LEGACY_SPEC_DIR/run.json" ]]; then
        # A run that wrote its recipe to the legacy nested
        # results/<hash>/runs/<run>/run.json path. The training scripts now
        # emit the flat results/<hash>/runs/<run>.json recipe (W8/e926665), so
        # this is a non-conforming run; refuse rather than silently promoting it.
        die "legacy nested run spec at $LEGACY_SPEC_DIR/run.json; expected the flat recipe at $SPEC_PATH. Re-run training with the updated scripts (which write the flat path) or move the recipe to $SPEC_PATH before re-running post_run.sh."
    else
        die "could not find run spec; expected $ARTIFACT_DIR/run.json or $SPEC_PATH"
    fi
    (cd "$SCRIPT_REPO_ROOT" && uv run --no-sync python -m json.tool "$SPEC_PATH" >/dev/null)
    run_post_run_contract "write" "$INPUT_SPEC_PATH"
fi

echo
echo "Issue comment template"
echo
echo "STATUS: post-training-run artifact/spec handoff prepared."
echo
echo "Run: \`$RUN_LABEL\`"
echo "Tracked spec: \`results/$HASH/runs/$RUN_LABEL.json\`"
echo "Bulk artifacts: \`_artifacts/$HASH/runs/$RUN_LABEL/\`"
echo
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "| Metric | Value |"
    echo "|---|---|"
    echo "| dry_run | metrics table would be rendered from \`training_summary.json\` |"
else
    render_metrics_table "$SUMMARY_PATH"
fi
echo
echo "Interpretation:"
echo "- Outcome: TODO(agent)"
echo "- Key metric movement: TODO(agent)"
echo "- Residuals/blockers: TODO(agent)"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_uv_lock_clean
fi

run_in_repo_or_print git add "$SPEC_PATH"
run_in_repo_or_print "$AGENT_COMMIT" --issue "$ISSUE" \
    -m "record post-run spec $RUN_LABEL"

BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
[[ "$BRANCH" != "HEAD" ]] || die "cannot submit auth request from detached HEAD"
AUTH_SPEC=$(
    cat <<EOF
Post-training-run protocol for \`$RUN_LABEL\`.

Intent:
- Preserve the tracked run spec at \`results/$HASH/runs/$RUN_LABEL.json\`.
- Keep bulk training outputs under \`_artifacts/$HASH/runs/$RUN_LABEL/\`.
- Record the metrics table from \`training_summary.json\` for issue follow-up.

Verification:
- Run spec JSON syntax verified before commit.
- Post-run provenance stamp recorded rlrmp/feedbax SHAs, schema versions, GraphSpec hashes where present, and pinned Feedbax manifest root.
- Feedbax TrainingRunManifest parity checked before commit when a matching manifest is present.
- Metrics-comment template rendered by \`scripts/post_run.sh\`.
EOF
)
run_in_repo_or_print "$MANDIBLE" auth request "$BRANCH" \
    --title "Record post-run spec $RUN_LABEL" \
    --issue "$ISSUE" \
    --spec "$AUTH_SPEC" \
    --no-watch

# Terminal run-status checkpoint (exactly one per run; phase=completed). This
# is the durable ledger marker a fresh session needs to tell a finished run
# from an in-flight one. Transient poll/retry detail stays in chat/nohup logs.
# See CLAUDE.md RunPod runbook §9 and issue e8b5b3b.
echo
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Run-status checkpoint (preview, not written)"
    echo "  command: $MANDIBLE issue checkpoint add $ISSUE --kind run-status --payload-file -"
    if [[ -f "$SUMMARY_PATH" ]]; then
        build_run_status_payload "completed" "$SUMMARY_PATH"
    else
        echo "  (training_summary.json absent; payload metrics_summary would be empty)"
        build_run_status_payload "completed" "$SUMMARY_PATH" 2>/dev/null || true
    fi
else
    echo "Emitting terminal run-status checkpoint (phase=completed) on $ISSUE"
    RUN_STATUS_PAYLOAD="$(build_run_status_payload "completed" "$SUMMARY_PATH")"
    printf '%s\n' "$RUN_STATUS_PAYLOAD" \
        | "$MANDIBLE" issue checkpoint add "$ISSUE" --kind run-status --payload-file -
fi

echo
echo "Checklist: decide whether this run requires coordination comments on c99ad9d (training-methods) and/or 4d38c15 (analyses)."
