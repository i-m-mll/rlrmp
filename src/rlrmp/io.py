"""File I/O helpers for rlrmp analysis scripts.

Provides utilities that analysis scripts use when writing output files to
``results/<exp>/notes/``. The primary helper here, ``update_marked_section``,
implements a *surgical in-place update* of auto-generated Markdown content
without disturbing hand-edited preambles or appended commentary.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Auto-generated section helpers
# ---------------------------------------------------------------------------

_MARKER_OPEN_PATTERN = r"<!-- AUTO-GENERATED: {name} -->"
_MARKER_CLOSE_PATTERN = r"<!-- /AUTO-GENERATED -->"

_MARKER_OPEN_RE = re.compile(r"<!-- AUTO-GENERATED: (\S+) -->")
_MARKER_CLOSE_LITERAL = "<!-- /AUTO-GENERATED -->"


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Write mappings to CSV using an explicit or first-row column contract."""

    columns = list(fieldnames if fieldnames is not None else (rows[0].keys() if rows else ()))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def json_ready(value: Any) -> Any:
    """Recursively convert array/scalar containers to JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def load_python_module(path: Path, *, module_name: str | None = None) -> Any:
    """Load a Python source file as a module with deterministic error handling."""

    name = module_name or f"rlrmp_dynamic_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_json_dumps(
    payload: Any,
    *,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Return stable compact machine JSON with one trailing newline."""

    return json.dumps(
        payload,
        default=default,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def write_compact_json(
    path: Path,
    payload: Any,
    *,
    default: Callable[[Any], Any] | None = None,
    atomic: bool = False,
) -> None:
    """Write stable compact machine JSON.

    Markdown remains the human-readable surface for generated interpretation;
    tracked JSON should be optimized for deterministic diffs and compact
    machine reads.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = compact_json_dumps(payload, default=default)
    if not atomic:
        path.write_text(encoded, encoding="utf-8")
        return

    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(encoded, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document."""

    return json.loads(path.read_text(encoding="utf-8"))


def _open_marker(name: str) -> str:
    return f"<!-- AUTO-GENERATED: {name} -->"


def _close_marker() -> str:
    return "<!-- /AUTO-GENERATED -->"


def _make_block(name: str, content: str) -> str:
    """Wrap ``content`` in open/close auto-generated markers."""
    # Ensure content ends with a single newline before the close marker.
    if not content.endswith("\n"):
        content = content + "\n"
    return f"{_open_marker(name)}\n{content}{_close_marker()}\n"


def update_marked_section(
    path: Path,
    marker_name: str,
    content: str,
) -> None:
    """Write ``content`` into a named auto-generated section in a Markdown file.

    The section is delimited by::

        <!-- AUTO-GENERATED: <marker_name> -->
        ... script-written content ...
        <!-- /AUTO-GENERATED -->

    Three cases:

    (i) **File does not exist** — creates the file with just the auto block.
        Any hand-edited preamble would be written by the human/agent *after*
        the initial script run, so this is the correct initialisation behaviour.

    (ii) **File exists, markers present** — replaces only the content between
        the open and close markers for ``marker_name``, preserving everything
        outside.

    (iii) **File exists, markers absent** — appends the auto block at the end
        of the file (preserving all prior content).

    Args:
        path: Destination Markdown file (e.g.
            ``results/<exp>/notes/variance_analysis.md``).
        marker_name: A short, stable identifier for this auto block, e.g.
            ``"variance_metrics"`` or ``"results_table"``. Must not contain
            spaces (use underscores).
        content: The Markdown content to place inside the markers. A trailing
            newline is added if absent.

    Note:
        Only one ``<!-- /AUTO-GENERATED -->`` close marker is used regardless
        of how many open markers exist, so ``marker_name`` identifies *which*
        block is targeted. If the file has multiple named blocks, each must
        pair with its own close marker.
    """
    if " " in marker_name:
        raise ValueError(
            f"marker_name must not contain spaces; got {marker_name!r}"
        )

    block = _make_block(marker_name, content)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block, encoding="utf-8")
        return

    text = path.read_text(encoding="utf-8")

    # Find the open marker for this specific name.
    open_marker = _open_marker(marker_name)
    open_pos = text.find(open_marker)

    if open_pos == -1:
        # Markers absent — append block at end.
        if not text.endswith("\n"):
            text = text + "\n"
        path.write_text(text + "\n" + block, encoding="utf-8")
        return

    # Markers present — find the matching close marker.
    close_pos = text.find(_close_marker(), open_pos + len(open_marker))
    if close_pos == -1:
        # Open marker exists but close marker is missing (malformed file).
        # Treat as "append after the open marker line" — replace from the open
        # marker to end-of-that-line, then append block.
        end_of_open_line = text.find("\n", open_pos)
        if end_of_open_line == -1:
            end_of_open_line = len(text)
        text = text[:open_pos] + block
        path.write_text(text, encoding="utf-8")
        return

    close_end = close_pos + len(_close_marker())
    # Skip any trailing newline after the close marker.
    if close_end < len(text) and text[close_end] == "\n":
        close_end += 1

    new_text = text[:open_pos] + block + text[close_end:]
    path.write_text(new_text, encoding="utf-8")
