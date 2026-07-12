"""Custody-backed declarative pieces for stabilization response figures."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from feedbax.contracts.figures import FigurePiece
from feedbax.contracts.manifest import ArtifactRef
from feedbax.plot.constructors import register_figure_piece


STABILIZATION_COMPARISON_IDS = (
    "pgd_1p05_stabilization_perturbation_responses",
    "pgd_ofb_budget_stabilization_perturbation_responses",
    "soft_pgd_stabilization_perturbation_responses",
)
STABILIZATION_PAYLOAD_ROLE = "stabilization_response_figure_payload"


def stabilization_piece_name(comparison_id: str) -> str:
    """Return the registered piece name for one stabilization comparison."""
    return f"rlrmp.stabilization_response.{comparison_id}"


def register_stabilization_figure_pieces(
    artifact_path: Path | str,
    *,
    replace: bool = True,
) -> None:
    """Register the three response comparisons from one custody payload."""
    path = Path(artifact_path).resolve()
    digest = sha256(path.read_bytes()).hexdigest()
    for comparison_id in STABILIZATION_COMPARISON_IDS:
        register_figure_piece(
            FigurePiece(
                name=stabilization_piece_name(comparison_id),
                description=f"Structured stabilization response payload for {comparison_id}.",
                artifact_ref=ArtifactRef(
                    role=STABILIZATION_PAYLOAD_ROLE,
                    logical_name=path.name,
                    sha256=digest,
                    media_type="application/json",
                    uri=str(path),
                ),
                data_path=f"comparisons.{comparison_id}",
                label=comparison_id,
                constructor="rlrmp.perturbation_response_traces",
            ),
            replace=replace,
        )


__all__ = [
    "STABILIZATION_COMPARISON_IDS",
    "STABILIZATION_PAYLOAD_ROLE",
    "register_stabilization_figure_pieces",
    "stabilization_piece_name",
]
