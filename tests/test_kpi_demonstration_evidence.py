from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[1]
M1_NOTE = REPO_ROOT / "results/2cb6a58/notes/engineering_smoke_evidence.md"
AUDIT_PACKET = REPO_ROOT / "results/509368b/notes/implementation_audit_packet.md"
LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
LONG_HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{41,}(?![0-9A-Fa-f])")


@dataclass(frozen=True)
class LifecycleEvidence:
    label: str
    row: str
    stop_run_set: str
    resume_run_set: str
    stop_transaction: str
    resume_transaction: str
    stop_loss: float
    resume_loss: float


LIFECYCLES = (
    LifecycleEvidence(
        "visible nominal",
        "force_visible__nominal_seed42_smoke100",
        "2026-07-13-6bae06ab",
        "2026-07-13-b5e80253",
        "tx-c29c9b098f364575a970f0f23ba889bf",
        "tx-3868327ebce5417aa8eeb169cb6d2cc8",
        76456.77734375,
        23244.3765625,
    ),
    LifecycleEvidence(
        "hidden nominal",
        "force_hidden__nominal_seed42_smoke100",
        "2026-07-13-1a170b75",
        "2026-07-13-1ac1bcee",
        "tx-f1f12722e4394a9388b8b6f586af7956",
        "tx-44f070e42cce42afb488669e95465c84",
        72068.46484375,
        22282.5359375,
    ),
    LifecycleEvidence(
        "visible broad PGD",
        "force_visible__broad_pgd_seed42_smoke100",
        "2026-07-13-7afcafb8",
        "2026-07-13-bd90c6fc",
        "tx-9716697ea7b541f5b6cdd32b01942a4e",
        "tx-41d81f97bb8447f097f906d0a9f094d9",
        71876.63203125,
        28446.34140625,
    ),
    LifecycleEvidence(
        "hidden broad PGD",
        "force_hidden__broad_pgd_seed42_smoke100",
        "2026-07-13-3d2417d7",
        "2026-07-13-43c9cd35",
        "tx-bdd90762159d41f3ba9249dd294bc5ab",
        "tx-4f269ed689754c039cb4c2f4e44a095c",
        73391.73828125,
        32210.2875,
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_blob_sha256(checkpoint_manifest: Path) -> str:
    payload = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    model_slot = next(slot for slot in payload["slots"] if slot["slot"] == "model")
    return model_slot["content_digest"]["blob_sha256"]


def _markdown_row(text: str, label: str) -> list[str]:
    prefix = f"| {label} |"
    row = next(line for line in text.splitlines() if line.startswith(prefix))
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _backtick_sha256(cell: str) -> list[str]:
    return [value for value in re.findall(r"`([^`]+)`", cell) if LOWER_SHA256.fullmatch(value)]


def _require_local_evidence() -> None:
    if not (REPO_ROOT / "_artifacts/orchestration").is_dir():
        pytest.skip("local-only 509 engineering-smoke artifacts are not materialized")


def test_governed_packet_sha256_values_are_full_lowercase_hex() -> None:
    for path in (M1_NOTE, AUDIT_PACKET):
        text = path.read_text(encoding="utf-8")
        candidates = LONG_HEX_TOKEN.findall(text)
        assert candidates, f"no full digest candidates found in {path}"
        malformed = [value for value in candidates if LOWER_SHA256.fullmatch(value) is None]
        assert malformed == [], f"malformed SHA-256 values in {path}: {malformed}"


def test_recorded_m1_lifecycle_hashes_and_wording_match_local_bytes() -> None:
    _require_local_evidence()
    note = M1_NOTE.read_text(encoding="utf-8")
    packet = AUDIT_PACKET.read_text(encoding="utf-8")

    for lifecycle in LIFECYCLES:
        model_hashes: list[str] = []
        recorded_row = _markdown_row(note, lifecycle.label)
        checkpoint_digests: list[str] = []
        manifest_digests: list[str] = []
        conformance_digests: list[str] = []
        for run_set, transaction, expected_batches, expected_loss in (
            (lifecycle.stop_run_set, lifecycle.stop_transaction, 50, lifecycle.stop_loss),
            (lifecycle.resume_run_set, lifecycle.resume_transaction, 100, lifecycle.resume_loss),
        ):
            run_set_root = REPO_ROOT / "_artifacts/orchestration" / run_set
            collected_manifest = run_set_root / "collected" / lifecycle.row / "manifest.json"
            checkpoint_manifest = (
                REPO_ROOT
                / "_artifacts/2cb6a58/runs"
                / lifecycle.row
                / "transactions"
                / transaction
                / "manifest.json"
            )
            conformance = run_set_root / "conformance.json"
            summary = run_set_root / "collected" / lifecycle.row / "training_summary.json"

            for byte_path in (collected_manifest, checkpoint_manifest, conformance, summary):
                assert byte_path.is_file(), byte_path

            manifest_digest = _sha256(collected_manifest)
            checkpoint_digest = _sha256(checkpoint_manifest)
            conformance_digest = _sha256(conformance)
            for digest in (manifest_digest, checkpoint_digest, conformance_digest):
                assert LOWER_SHA256.fullmatch(digest)
            manifest_digests.append(manifest_digest)
            checkpoint_digests.append(checkpoint_digest)
            conformance_digests.append(conformance_digest)

            manifest_payload = json.loads(collected_manifest.read_text(encoding="utf-8"))
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            assert manifest_payload["completed_batches"] == expected_batches
            assert summary_payload["metrics"]["train_loss"] == expected_loss
            model_hashes.append(_model_blob_sha256(checkpoint_manifest))

        assert model_hashes[0] != model_hashes[1]
        assert _backtick_sha256(recorded_row[3]) == checkpoint_digests
        assert _backtick_sha256(recorded_row[4]) == manifest_digests
        assert _backtick_sha256(recorded_row[5]) == conformance_digests

        stop_registration = json.loads(
            (
                REPO_ROOT
                / "_artifacts/orchestration"
                / lifecycle.stop_run_set
                / "registration.json"
            ).read_text(encoding="utf-8")
        )
        stop_manifest = json.loads(
            (
                REPO_ROOT
                / "_artifacts/orchestration"
                / lifecycle.stop_run_set
                / "collected"
                / lifecycle.row
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert stop_manifest["status"] == "cancelled"
        assert stop_manifest["completed_batches"] == 50
        assert stop_registration["status"] == "stopped"
        assert stop_registration["certificate_overall"] == "pass"

    visible_nominal_digest = "36412dcf4db037094151f506afa9c2c86d24e9fae91b9bcb6e7fb34cefd6ea5a"
    assert visible_nominal_digest in note
    packet_manifest_section = packet.split(
        "The completed visible-nominal `TrainingRunManifest`", maxsplit=1
    )[1].split("Stop-state wording is exact", maxsplit=1)[0]
    assert LONG_HEX_TOKEN.findall(packet_manifest_section) == [visible_nominal_digest]

    for family, matrix_path in (
        ("M1", REPO_ROOT / "results/2cb6a58/runs/matrix.json"),
        ("A1", REPO_ROOT / "results/4eb51ee/runs/matrix.json"),
    ):
        packet_row = _markdown_row(packet, family)
        assert _backtick_sha256(packet_row[1]) == [_sha256(matrix_path)]

    lock_digest = _sha256(REPO_ROOT / "uv.lock")
    assert f"| `uv.lock` SHA-256 | `{lock_digest}` |" in note
    assert f"| `uv.lock` SHA-256 | `{lock_digest}` |" in packet
    assert "status=cancelled" in note and "completed_batches=50" in note
    assert "status=stopped" in note
    assert "status=cancelled" in packet and "completed_batches=50" in packet
    assert "status=stopped" in packet
