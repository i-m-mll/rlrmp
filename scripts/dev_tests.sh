#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Guard (issue 81e4588): the shared venv's editable rlrmp install can point at
# a stale worktree (e.g. a deleted integration worktree), which silently
# imports and tests the wrong source tree even with PYTHONPATH set correctly,
# if something upstream of PYTHONPATH on sys.path wins. Resolve where `rlrmp`
# would actually import from and fail loudly on any mismatch, rather than
# letting tests run silently against the wrong tree.
EXPECTED_RLRMP_SRC="$REPO_ROOT/src"
if ! GUARD_OUTPUT="$(uv run --no-sync python -c '
import pathlib, sys
expected = pathlib.Path(sys.argv[1]).resolve() / "rlrmp"
import rlrmp
resolved = pathlib.Path(rlrmp.__file__).resolve().parent
if resolved != expected:
    print(f"resolved={resolved} expected={expected}")
    sys.exit(1)
' "$EXPECTED_RLRMP_SRC" 2>&1)"; then
    echo "ERROR: scripts/dev_tests.sh refused to run: stale editable install detected." >&2
    echo "The rlrmp package the shared venv would import does not match this worktree's src/." >&2
    echo "$GUARD_OUTPUT" >&2
    echo "Expected: $EXPECTED_RLRMP_SRC/rlrmp" >&2
    echo "Fix: run 'uv sync' from the repo root that should own the editable install, or set PYTHONPATH to point at this worktree's src/ (see issue 81e4588)." >&2
    exit 1
fi

exec uv run --no-sync python "$REPO_ROOT/scripts/dev_tests.py" "$@"
