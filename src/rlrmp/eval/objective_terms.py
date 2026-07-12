"""Cached objective-term evaluation contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OBJECTIVE_TERMS_EVALUATION_TYPE = "rlrmp.eval.gru_diagnostics"
OBJECTIVE_TERM_NAMES = (
    "running_state_q",
    "terminal_state_q_f",
    "command_r",
    "force_filter_state",
    "disturbance_integrator_state",
    "total",
)


def objective_term_rows(states: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize objective summaries already held by an evaluation manifest.

    This boundary intentionally accepts only cached summaries. Rollout execution,
    checkpoint loading, and filesystem path resolution belong to upstream
    registered evaluation recipes.
    """

    payload = states.get("rows", states.get("runs", ()))
    values = payload.values() if isinstance(payload, Mapping) else payload
    rows: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        run_id = value.get("run_id", value.get("id"))
        if run_id is None:
            continue
        terms = value.get("terms", value.get("objective_terms", {}))
        references = value.get("reference_terms", value.get("extlqg_terms", {}))
        rows.append(
            {
                "run_id": str(run_id),
                "checkpoint_selection": value.get("checkpoint_selection"),
                "bank": value.get("bank", value.get("split_bank")),
                "terms": _known_terms(terms),
                "reference_terms": _known_terms(references),
            }
        )
    return rows


def _known_terms(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {name: value[name] for name in OBJECTIVE_TERM_NAMES if name in value}


__all__ = ["OBJECTIVE_TERM_NAMES", "OBJECTIVE_TERMS_EVALUATION_TYPE", "objective_term_rows"]
