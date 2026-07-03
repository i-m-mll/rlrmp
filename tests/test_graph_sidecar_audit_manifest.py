"""Guard the committed graph-sidecar fidelity audit manifest (issue e9fc384).

Issue [ae15851] will convert the tracked archived Feedbax `model.graph.json`
sidecars into clean new-format `GraphSpec` loadability regression fixtures. It
must consume the audit manifest produced here
(`results/e9fc384/notes/graph_sidecar_audit_manifest.json`), not a free glob,
so that fixture conversion cannot silently pick up a new, unaudited sidecar.
This test is the drift guard: it fails on an empty audited set, a live count
that disagrees with the manifest, a missing path/hash, a hash mismatch, a
stale manifest entry with no matching live file, or a live sidecar that is
not present in the manifest at all.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "results" / "e9fc384" / "notes" / "graph_sidecar_audit_manifest.json"
SIDECAR_PATTERNS = ("results/**/model.graph.json", "results/**/*.graph.json")
CONVERTED_FIXTURE_PREFIXES = ("results/ae15851/converted/",)


def _live_sidecar_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *SIDECAR_PATTERNS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        {
            line
            for line in result.stdout.splitlines()
            if line and not line.startswith(CONVERTED_FIXTURE_PREFIXES)
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"manifest not found at {MANIFEST_PATH}"
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_manifest_schema_version_and_nonempty_file_set() -> None:
    manifest = _load_manifest()

    assert manifest.get("schema_version") == 1
    files = manifest.get("files")
    assert isinstance(files, list)
    assert len(files) > 0, "audited file set must not be empty"
    assert manifest.get("audited_count") == len(files)


def test_manifest_audited_count_matches_live_git_ls_files() -> None:
    manifest = _load_manifest()
    live_paths = _live_sidecar_paths()

    assert manifest["audited_count"] == len(live_paths), (
        f"manifest audited_count={manifest['audited_count']} disagrees with live "
        f"git ls-files count={len(live_paths)}; re-run "
        "results/e9fc384/scripts/build_graph_sidecar_audit_manifest.py and update the "
        "audit narrative if the tracked sidecar set has legitimately changed"
    )


def test_manifest_entries_have_path_and_hash() -> None:
    manifest = _load_manifest()

    missing: list[str] = []
    for entry in manifest["files"]:
        path = entry.get("path")
        sha256 = entry.get("sha256")
        if not path:
            missing.append(f"<entry missing path>: {entry!r}")
            continue
        if not sha256:
            missing.append(f"{path}: missing sha256")

    assert missing == [], f"manifest entries with missing path/hash: {missing}"


def test_manifest_hashes_match_live_file_content() -> None:
    manifest = _load_manifest()

    mismatches: list[str] = []
    for entry in manifest["files"]:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            # Covered by test_manifest_has_no_stale_entries; skip hashing here.
            continue
        live_hash = _sha256(path)
        if live_hash != entry["sha256"]:
            mismatches.append(
                f"{entry['path']}: manifest sha256={entry['sha256']} live sha256={live_hash}"
            )

    assert mismatches == [], f"hash mismatches (sidecar changed since audit): {mismatches}"


def test_manifest_has_no_stale_entries() -> None:
    """Every manifest path must correspond to a currently tracked sidecar."""
    manifest = _load_manifest()
    live_paths = set(_live_sidecar_paths())

    stale = sorted(entry["path"] for entry in manifest["files"] if entry["path"] not in live_paths)

    assert stale == [], f"manifest entries with no matching live tracked sidecar: {stale}"


def test_no_unaudited_new_sidecars() -> None:
    """Every live tracked sidecar must be present in the manifest."""
    manifest = _load_manifest()
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    live_paths = set(_live_sidecar_paths())

    unaudited = sorted(live_paths - manifest_paths)

    assert unaudited == [], (
        f"new tracked graph sidecar(s) matching {SIDECAR_PATTERNS} are not in the audit "
        f"manifest: {unaudited}. Re-run "
        "results/e9fc384/scripts/build_graph_sidecar_audit_manifest.py, classify the new "
        "sidecar(s), and update results/e9fc384/notes/graph_sidecar_audit.md before "
        "issue ae15851 may treat them as loadability fixtures."
    )


def test_manifest_classification_summary_is_consistent() -> None:
    manifest = _load_manifest()
    files = manifest["files"]

    valid_classifications = {"clean", "known_wrong"}
    for entry in files:
        assert entry["classification"] in valid_classifications, entry["path"]

    summary = manifest["classification_summary"]
    for classification in valid_classifications:
        expected = sum(1 for entry in files if entry["classification"] == classification)
        assert summary[classification] == expected, classification


def test_known_wrong_entries_carry_conversion_candidate_key() -> None:
    """The (retired type, structural predicate, metadata.version, hash) key must be present."""
    manifest = _load_manifest()

    for entry in manifest["files"]:
        key = entry.get("conversion_candidate_key")
        assert isinstance(key, dict), entry["path"]
        assert "retired_type" in key
        assert "structural_predicate" in key
        assert key.get("metadata_version") == entry.get("metadata_version")
        assert key.get("audited_fixture_hash") == entry.get("sha256")


def test_cs_30f2313_sidecars_are_known_wrong_cs_lss_candidates() -> None:
    """Regression guard for the specific finding this audit issue was filed to confirm."""
    manifest = _load_manifest()
    by_path = {entry["path"]: entry for entry in manifest["files"]}

    for path in (
        "results/30f2313/runs/cs_stochastic_gru__hidden_penalty/model.graph.json",
        "results/30f2313/runs/cs_stochastic_gru__no_hidden_penalty/model.graph.json",
    ):
        assert path in by_path, path
        entry = by_path[path]
        assert entry["classification"] == "known_wrong", path
        assert entry["expected_conversion_family"] == "cs_lss", path
        assert entry["structural_family_actual"] == "point_mass", path
        assert "LinearStateSpace" not in entry["structural_types"], path


def test_baseline_vrnn_sidecars_are_clean_vanilla_rnn_family() -> None:
    manifest = _load_manifest()
    by_path = {entry["path"]: entry for entry in manifest["files"]}

    vrnn_paths = [
        f"results/b41c940/migrated/efc4d68/baseline_vrnn__{suffix}/model.graph.json"
        for suffix in ("jerk", "none", "smooth", "smooth_jerk")
    ]
    for path in vrnn_paths:
        assert path in by_path, path
        entry = by_path[path]
        assert entry["classification"] == "clean", path
        assert entry["controller_kind"] == "vanilla_rnn", path
        assert entry["structural_subfamily_actual"] == "point_mass_vanilla_rnn", path
