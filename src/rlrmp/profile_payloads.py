"""Governed profile payloads for declarative figures."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from feedbax.contracts.figures import FigurePiece
from feedbax.contracts.manifest import ArtifactRef, StrictModel
from feedbax.plot.constructors import register_figure_piece
from pydantic import model_validator

from rlrmp.paths import REPO_ROOT


PROFILE_PAYLOAD_SPEC_SCHEMA_ID = "rlrmp.spec.figure_profile_payload"
PROFILE_PAYLOAD_SPEC_SCHEMA_VERSION = "rlrmp.spec.figure_profile_payload.v1"
PROFILE_PAYLOAD_SCHEMA_ID = "rlrmp.figure_data.profile_comparison"
PROFILE_PAYLOAD_SCHEMA_VERSION = "rlrmp.figure_data.profile_comparison.v1"


class ProfileSeriesColumns(StrictModel):
    """Column bindings and presentation metadata for one profile series."""

    label: str
    mean_column: str
    spread_column: str
    spread_kind: Literal["sd", "sem"]
    color: str
    line_dash: str = "solid"


class ProfilePayloadSpec(StrictModel):
    """Authored adoption spec for an upstream profile CSV and summary."""

    schema_id: str = PROFILE_PAYLOAD_SPEC_SCHEMA_ID
    schema_version: str = PROFILE_PAYLOAD_SPEC_SCHEMA_VERSION
    piece_name: str
    issue: str
    topic: str
    condition: str
    profile_key: str = "forward_velocity"
    source_csv: ArtifactRef
    source_summary: ArtifactRef
    output: ArtifactRef
    time_column: str = "time_s"
    series: list[ProfileSeriesColumns]

    @model_validator(mode="after")
    def _validate_schema_and_series(self) -> "ProfilePayloadSpec":
        if self.schema_id != PROFILE_PAYLOAD_SPEC_SCHEMA_ID:
            raise ValueError(f"unsupported profile payload schema_id: {self.schema_id!r}")
        if self.schema_version != PROFILE_PAYLOAD_SPEC_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile payload schema_version: {self.schema_version!r}")
        labels = [series.label for series in self.series]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("profile payload series labels must be non-empty and unique")
        return self


def load_profile_payload_spec(path: Path | str) -> ProfilePayloadSpec:
    """Load one governed profile-payload adoption spec."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProfilePayloadSpec.model_validate(payload)


def materialize_profile_payload(
    spec: ProfilePayloadSpec,
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Adopt a wide profile CSV into the registered figure-payload schema."""

    csv_path = _resolve_uri(spec.source_csv, repo_root)
    summary_path = _resolve_uri(spec.source_summary, repo_root)
    _verify_artifact(csv_path, spec.source_csv)
    _verify_artifact(summary_path, spec.source_summary)

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Profile CSV is empty: {csv_path}")

    required = {
        spec.time_column,
        *(
            column
            for series in spec.series
            for column in (series.mean_column, series.spread_column)
        ),
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Profile CSV {csv_path} is missing columns: {missing}")

    time = _float_column(rows, spec.time_column)
    series_payload = []
    for series in spec.series:
        mean = _float_column(rows, series.mean_column)
        spread = _float_column(rows, series.spread_column)
        series_payload.append(
            {
                "label": series.label,
                "color": series.color,
                "line_dash": series.line_dash,
                "spread_kind": series.spread_kind,
                "profile": {
                    "time": time,
                    "mean": mean,
                    "lower": [value - delta for value, delta in zip(mean, spread, strict=True)],
                    "upper": [value + delta for value, delta in zip(mean, spread, strict=True)],
                },
            }
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "schema_id": PROFILE_PAYLOAD_SCHEMA_ID,
        "schema_version": PROFILE_PAYLOAD_SCHEMA_VERSION,
        "issue": spec.issue,
        "topic": spec.topic,
        "condition": spec.condition,
        "cell": {
            "run_id": f"{spec.issue}:{spec.topic}",
            "display_name": spec.condition,
            spec.profile_key: {"series": series_payload},
        },
        "summary": summary,
        "source_artifacts": [
            spec.source_csv.model_dump(mode="json", exclude_none=True),
            spec.source_summary.model_dump(mode="json", exclude_none=True),
        ],
    }
    destination = output_path or _resolve_uri(spec.output, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _verify_artifact(destination, spec.output)
    return payload


def register_profile_payload_piece(
    spec: ProfilePayloadSpec,
    *,
    repo_root: Path = REPO_ROOT,
    replace: bool = True,
) -> None:
    """Register one profile payload as a reusable declarative figure piece."""

    output_path = _resolve_uri(spec.output, repo_root)
    artifact = spec.output.model_copy(
        update={
            "uri": str(output_path.resolve()),
            "sha256": _sha256(output_path) if output_path.is_file() else spec.output.sha256,
            "size_bytes": output_path.stat().st_size if output_path.is_file() else None,
        }
    )
    register_figure_piece(
        FigurePiece(
            name=spec.piece_name,
            description=(
                f"Governed {spec.condition} profile payload for {spec.issue}/{spec.topic}."
            ),
            artifact_ref=artifact,
            data_path=None,
            label=spec.condition,
            constructor="rlrmp.profile_series",
        ),
        replace=replace,
    )


def register_profile_stock_pieces(*, repo_root: Path = REPO_ROOT, replace: bool = True) -> None:
    """Register every migrated analytical/nominal stock profile piece."""

    for path in profile_stock_spec_paths(repo_root):
        register_profile_payload_piece(
            load_profile_payload_spec(path), repo_root=repo_root, replace=replace
        )


def profile_stock_spec_paths(repo_root: Path = REPO_ROOT) -> tuple[Path, ...]:
    """Return the governed payload specs for this migration wave."""

    return tuple(
        repo_root / "results" / issue / "notes" / "profile_payload_regeneration_spec.json"
        for issue in ("376d023", "a378b34", "e148f33")
    )


def _resolve_uri(artifact: ArtifactRef, root: Path) -> Path:
    if artifact.uri is None:
        raise ValueError(f"Artifact {artifact.logical_name!r} has no URI")
    path = Path(artifact.uri)
    return path if path.is_absolute() else root / path


def _verify_artifact(path: Path, artifact: ArtifactRef) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if artifact.sha256 is not None and _sha256(path) != artifact.sha256:
        raise ValueError(f"Artifact hash mismatch for {path}")


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
