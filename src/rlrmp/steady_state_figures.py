"""Custody-backed figure pieces for steady-state perturbation responses."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from feedbax.contracts.figures import FigurePiece
from feedbax.contracts.manifest import ArtifactRef
from feedbax.plot.constructors import register_figure_piece

from rlrmp.paths import REPO_ROOT


STEADY_STATE_DETAIL_PATH = Path(
    "_artifacts/87424a4/notes/steady_state_perturbation_bank_detail.json"
)
STEADY_STATE_COMPARISON_IDS = (
    "delayed_sisu_effective_020a65b",
    "matched_020a65b_h0_no_pgd_vs_pgd",
    "matched_020a65b_no_pgd_vs_pgd",
    "undelayed_targetfix_sisu_effective_020a65b",
)


def steady_state_piece_name(comparison_id: str) -> str:
    """Return the registered piece name for one governed comparison."""
    return f"rlrmp.steady_state_87424a4.{comparison_id}"


def register_steady_state_figure_pieces(
    *,
    artifact_path: Path | str | None = None,
    replace: bool = True,
) -> None:
    """Register all surviving 87424a4 comparisons from custody JSON."""
    path = Path(artifact_path or (REPO_ROOT / STEADY_STATE_DETAIL_PATH)).resolve()
    if artifact_path is None and not path.exists():
        return
    digest = sha256(path.read_bytes()).hexdigest()
    for comparison_id in STEADY_STATE_COMPARISON_IDS:
        register_figure_piece(
            FigurePiece(
                name=steady_state_piece_name(comparison_id),
                description=f"Structured steady-state response payload for {comparison_id}.",
                artifact_ref=ArtifactRef(
                    role="rlrmp-steady-state-perturbation-detail",
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
    "STEADY_STATE_COMPARISON_IDS",
    "STEADY_STATE_DETAIL_PATH",
    "register_steady_state_figure_pieces",
    "steady_state_piece_name",
]
