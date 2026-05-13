"""Unit tests for `rlrmp.io.update_marked_section`.

Covers the three cases documented in the function docstring:
  (i) no existing file → creates with markers
  (ii) existing file with markers → only replaces marker content
  (iii) existing file without markers → appends marker block at the end
"""

from __future__ import annotations

import pytest

from rlrmp.io import update_marked_section


# ---------------------------------------------------------------------------
# Case (i): no existing file
# ---------------------------------------------------------------------------


def test_creates_file_when_absent(tmp_path):
    """If the file does not exist, update_marked_section creates it."""
    p = tmp_path / "notes" / "analysis.md"
    update_marked_section(p, "results_table", "| A | B |\n|---|---|\n| 1 | 2 |")

    assert p.exists()
    text = p.read_text()
    assert "<!-- AUTO-GENERATED: results_table -->" in text
    assert "<!-- /AUTO-GENERATED -->" in text
    assert "| A | B |" in text


def test_creates_parent_dirs_when_absent(tmp_path):
    """Missing parent directories are created automatically."""
    p = tmp_path / "deep" / "nested" / "notes.md"
    update_marked_section(p, "intro", "hello")
    assert p.exists()


# ---------------------------------------------------------------------------
# Case (ii): existing file with markers → only replaces marker content
# ---------------------------------------------------------------------------


def test_replaces_marker_content_preserves_preamble(tmp_path):
    """Hand-written preamble before the auto block is preserved on re-run."""
    p = tmp_path / "analysis.md"
    preamble = "# My Analysis\n\nCorrected after go-cue alignment fix (Bug: 06f7faf).\n\n"
    initial_auto = "<!-- AUTO-GENERATED: metrics -->\nold content\n<!-- /AUTO-GENERATED -->\n"
    p.write_text(preamble + initial_auto)

    update_marked_section(p, "metrics", "new content")

    text = p.read_text()
    assert text.startswith(preamble)
    assert "old content" not in text
    assert "new content" in text
    assert "<!-- AUTO-GENERATED: metrics -->" in text
    assert "<!-- /AUTO-GENERATED -->" in text


def test_replaces_marker_content_preserves_postamble(tmp_path):
    """Content after the auto block (e.g. a follow-up note) is preserved."""
    p = tmp_path / "analysis.md"
    initial = (
        "# Header\n\n"
        "<!-- AUTO-GENERATED: table -->\nstale table\n<!-- /AUTO-GENERATED -->\n"
        "\n## Notes\nSome hand-written notes.\n"
    )
    p.write_text(initial)

    update_marked_section(p, "table", "fresh table")

    text = p.read_text()
    assert "stale table" not in text
    assert "fresh table" in text
    assert "## Notes\nSome hand-written notes." in text


def test_replaces_only_the_named_block(tmp_path):
    """When multiple named blocks exist, only the targeted one is updated."""
    p = tmp_path / "multi.md"
    initial = (
        "<!-- AUTO-GENERATED: block_a -->\ncontent A\n<!-- /AUTO-GENERATED -->\n"
        "\n"
        "<!-- AUTO-GENERATED: block_b -->\ncontent B\n<!-- /AUTO-GENERATED -->\n"
    )
    p.write_text(initial)

    update_marked_section(p, "block_a", "new A")

    text = p.read_text()
    assert "new A" in text
    assert "content A" not in text
    assert "content B" in text  # block_b untouched


# ---------------------------------------------------------------------------
# Case (iii): existing file without markers → appends block
# ---------------------------------------------------------------------------


def test_appends_block_when_markers_absent(tmp_path):
    """If the file exists but has no markers, the block is appended."""
    p = tmp_path / "notes.md"
    existing = "# Existing content\n\nSome prose here.\n"
    p.write_text(existing)

    update_marked_section(p, "auto_section", "generated stuff")

    text = p.read_text()
    assert text.startswith(existing)
    assert "<!-- AUTO-GENERATED: auto_section -->" in text
    assert "generated stuff" in text
    assert "<!-- /AUTO-GENERATED -->" in text


def test_appends_with_separator_when_missing_trailing_newline(tmp_path):
    """Existing file without trailing newline should not merge with appended block."""
    p = tmp_path / "notes.md"
    p.write_text("no trailing newline")

    update_marked_section(p, "block", "content")

    text = p.read_text()
    # The block must start on its own line
    assert "\n<!-- AUTO-GENERATED: block -->" in text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_raises_on_marker_name_with_spaces(tmp_path):
    """marker_name must not contain spaces."""
    p = tmp_path / "notes.md"
    with pytest.raises(ValueError, match="marker_name must not contain spaces"):
        update_marked_section(p, "bad name here", "content")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_repeated_run(tmp_path):
    """Running update_marked_section twice with the same content is idempotent."""
    p = tmp_path / "notes.md"
    content = "row 1\nrow 2\n"

    update_marked_section(p, "section", content)
    text_first = p.read_text()

    update_marked_section(p, "section", content)
    text_second = p.read_text()

    assert text_first == text_second
