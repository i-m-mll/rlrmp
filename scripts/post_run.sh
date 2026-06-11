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

echo "Post-run protocol"
echo "  issue: $ISSUE"
echo "  run: $RUN_LABEL"
echo "  run spec: $SPEC_PATH"
echo "  artifacts: $ARTIFACT_DIR"

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_clean_index
fi

case "$ARTIFACTS_SRC" in
    "")
        echo "Sync: using existing local artifact/spec paths."
        ;;
    local:*)
        SOURCE_DIR="${ARTIFACTS_SRC#local:}"
        [[ -d "$SOURCE_DIR" ]] || die "local artifact source does not exist: $SOURCE_DIR"
        run_or_print mkdir -p "$ARTIFACT_DIR"
        run_or_print rsync -a "$SOURCE_DIR"/ "$ARTIFACT_DIR"/
        ;;
    modal|modal:*)
        VOLUME_NAME="$MODAL_VOLUME_DEFAULT"
        if [[ "$ARTIFACTS_SRC" == modal:* && "$ARTIFACTS_SRC" != "modal:" ]]; then
            VOLUME_NAME="${ARTIFACTS_SRC#modal:}"
        fi
        run_or_print mkdir -p "$RUNS_DIR" "$ARTIFACT_DIR"
        run_or_print modal volume get --force "$VOLUME_NAME" \
            "results/$HASH/runs/$RUN_LABEL" "$RUNS_DIR"
        run_or_print modal volume get --force "$VOLUME_NAME" \
            "_artifacts/$HASH/runs/$RUN_LABEL" "$ARTIFACT_DIR/.."
        ;;
    pod:*)
        RSYNC_SOURCE="${ARTIFACTS_SRC#pod:}"
        [[ -n "$RSYNC_SOURCE" ]] || die "pod source must include an rsync source"
        run_or_print mkdir -p "$ARTIFACT_DIR"
        run_or_print rsync -az --stats --no-owner --no-group \
            "$RSYNC_SOURCE"/ "$ARTIFACT_DIR"/
        ;;
    *)
        if [[ -d "$ARTIFACTS_SRC" ]]; then
            run_or_print mkdir -p "$ARTIFACT_DIR"
            run_or_print rsync -a "$ARTIFACTS_SRC"/ "$ARTIFACT_DIR"/
        else
            die "unknown --artifacts-src form or missing directory: $ARTIFACTS_SRC"
        fi
        ;;
esac

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: would write/verify tracked run spec at $SPEC_PATH"
    echo "DRY-RUN: would render metrics table from $SUMMARY_PATH"
    echo "DRY-RUN: would git add $SPEC_PATH before agent-commit"
else
    mkdir -p "$RUNS_DIR"
    if [[ -f "$ARTIFACT_DIR/run.json" ]]; then
        cp "$ARTIFACT_DIR/run.json" "$SPEC_PATH"
    elif [[ -f "$LEGACY_SPEC_DIR/run.json" ]]; then
        cp "$LEGACY_SPEC_DIR/run.json" "$SPEC_PATH"
    elif [[ -f "$SPEC_PATH" ]]; then
        :
    else
        die "could not find run spec; expected $ARTIFACT_DIR/run.json or $LEGACY_SPEC_DIR/run.json"
    fi
    (cd "$SCRIPT_REPO_ROOT" && uv run --no-sync python -m json.tool "$SPEC_PATH" >/dev/null)
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
- Metrics-comment template rendered by \`scripts/post_run.sh\`.
EOF
)
run_in_repo_or_print "$MANDIBLE" auth request "$BRANCH" \
    --title "Record post-run spec $RUN_LABEL" \
    --issue "$ISSUE" \
    --spec "$AUTH_SPEC" \
    --no-watch

echo
echo "Checklist: decide whether this run requires coordination comments on c99ad9d (training-methods) and/or 4d38c15 (analyses)."
