#!/usr/bin/env bash
# Opt-in migration helper for renaming pre-Phase-2 single-underscore run-IDs
# to the canonical `<group>__<variant>` double-underscore form.
#
# Bug: 0077b42 — Phase 2 completion: structure migration deferred items.
#
# Usage:
#   ./scripts/migrate_run_ids.sh                  # dry-run, _artifacts/, all exps
#   ./scripts/migrate_run_ids.sh --apply          # actually rename
#   ./scripts/migrate_run_ids.sh --exp part2_5    # restrict to one experiment
#   ./scripts/migrate_run_ids.sh --root results   # walk results/ instead of _artifacts/
#   ./scripts/migrate_run_ids.sh --apply --root results --exp part2_5
#
# By default the script runs in dry-run mode (no disk writes) and prints
# every proposed rename. Pass --apply to actually `mv` directories.
#
# The rename table is a hard-coded set of `old -> new` pairs taken from
# `_artifacts/README.md` ("Run-ID naming: old → new"). Directories whose
# name does not appear in the table — or whose name already contains `__` —
# are left untouched.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

APPLY=0
ROOT="_artifacts"
EXP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply)
            APPLY=1
            shift
            ;;
        --root)
            ROOT="$2"
            shift 2
            ;;
        --exp)
            EXP="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# Resolve repo root from this script's location: scripts/ is one level under repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WALK_ROOT="$REPO_ROOT/$ROOT"
if [[ -n "$EXP" ]]; then
    WALK_ROOT="$WALK_ROOT/$EXP"
fi

if [[ ! -d "$WALK_ROOT" ]]; then
    echo "Walk root does not exist: $WALK_ROOT" >&2
    echo "(Nothing to migrate. Pass --root and/or --exp to narrow the walk.)" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Rename table
# ---------------------------------------------------------------------------
# Each entry is "old_name new_name". Names are matched against the basename
# of every directory under $WALK_ROOT (any depth).

read -r -d '' RENAMES <<'EOF' || true
running_cost_standard running_cost__standard
softmin_standard softmin__standard
default_standard default__standard
combined_standard combined__standard
running_cost_cvar running_cost__cvar
running_cost_nn1e4 running_cost__nn1e4
running_cost_nn1e6 running_cost__nn1e6
baseline_standard_12k baseline__standard_12k
baseline_apt baseline__apt
baseline_cvar baseline__cvar
baseline_no_pert baseline__no_pert
baseline_nn1e6 baseline__nn1e6
apt_lr001 apt__lr001
apt_pert2 apt__pert2
tier1_redo tier1__redo
ratio_sweep ratio__sweep
mult_pop5 mult__pop5
mult_single mult__single
vanilla_pop5 vanilla__pop5
vanilla_single vanilla__single
minimax_single minimax__single
ratio03_pop5 ratio03__pop5
ratio03_single ratio03__single
EOF

# ---------------------------------------------------------------------------
# Walk and propose renames
# ---------------------------------------------------------------------------

if [[ "$APPLY" -eq 1 ]]; then
    echo "MODE: APPLY — directories will be renamed in place."
else
    echo "MODE: DRY-RUN — no disk writes (pass --apply to rename)."
fi
echo "Walking: $WALK_ROOT"
echo

count=0
applied=0
skipped_dest_exists=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    OLD="${line%% *}"
    NEW="${line##* }"
    [[ "$OLD" == "$NEW" ]] && continue

    # Find every directory whose basename equals OLD anywhere under WALK_ROOT.
    # -depth ensures children are processed before parents (safer for nested matches).
    while IFS= read -r dir; do
        # Skip names that already contain __ (already migrated).
        base="$(basename "$dir")"
        [[ "$base" == *"__"* ]] && continue
        [[ "$base" != "$OLD" ]] && continue

        new_path="$(dirname "$dir")/$NEW"
        count=$((count + 1))

        if [[ -e "$new_path" ]]; then
            echo "SKIP   $dir"
            echo "       -> $new_path  (destination already exists)"
            skipped_dest_exists=$((skipped_dest_exists + 1))
            continue
        fi

        echo "RENAME $dir"
        echo "       -> $new_path"

        if [[ "$APPLY" -eq 1 ]]; then
            mv -- "$dir" "$new_path"
            applied=$((applied + 1))
        fi
    done < <(find "$WALK_ROOT" -depth -type d -name "$OLD" 2>/dev/null)
done <<< "$RENAMES"

echo
echo "Summary: $count proposed rename(s); $applied applied; $skipped_dest_exists skipped (destination existed)."
if [[ "$APPLY" -ne 1 && "$count" -gt 0 ]]; then
    echo "Re-run with --apply to actually rename."
fi
