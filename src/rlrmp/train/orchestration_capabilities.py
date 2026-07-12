"""Behavioral probes for the in-flight Feedbax Lane-0 orchestration fixes."""

from __future__ import annotations

SCHEDULED_PREFLIGHT_SKIP_REASON = (
    "installed Feedbax lacks Lane-0 controller_optimizer preflight/context discovery"
)
SCHEDULED_CERTIFY_SKIP_REASON = (
    "installed Feedbax lacks Lane-0 controller_optimizer CERTIFY discovery"
)


def scheduled_preflight_capable() -> bool:
    """Return whether scheduled controller optimizers and metadata contexts are found."""
    from feedbax.orchestration import schedule_eval, stages

    payload = {
        "method_payload": {"payload": {"controller_optimizer": {"type": "adam", "params": {}}}},
        "metadata": {
            "resume_context": _context(),
            "optimizer_build_context": _context(),
        },
    }
    try:
        optimizers = stages._optimizer_payloads(payload)
        resume = schedule_eval.extract_resume_context(payload)
        build = schedule_eval.extract_optimizer_build_context(payload)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(optimizers) and resume == _context() and build == _context()


def scheduled_certify_capable() -> bool:
    """Return whether CERTIFY discovers the governed controller optimizer path."""
    from feedbax.orchestration import conformance

    payload = {
        "method_payload": {"payload": {"controller_optimizer": {"type": "adam", "params": {}}}}
    }
    row = conformance.ConformanceRowArtifacts(
        row_id="controller-optimizer-capability-probe",
        bundle_row_spec=payload,
        training_diagnostics=None,
        manifest_payload=None,
    )
    try:
        discovered = conformance._optimizer_spec_payload(row)
    except (AttributeError, TypeError, ValueError):
        return False
    return isinstance(discovered, dict) and discovered.get("type") == "adam"


def missing_scheduled_capability_reasons() -> tuple[str, ...]:
    """Return only the named scheduled-path capabilities absent from Feedbax."""
    reasons = []
    if not scheduled_preflight_capable():
        reasons.append(SCHEDULED_PREFLIGHT_SKIP_REASON)
    if not scheduled_certify_capable():
        reasons.append(SCHEDULED_CERTIFY_SKIP_REASON)
    return tuple(reasons)


def _context() -> dict[str, int]:
    return {
        "schedule_origin_step": 0,
        "current_step": 12,
        "optimizer_count_at_current_step": 0,
    }


__all__ = [
    "SCHEDULED_CERTIFY_SKIP_REASON",
    "SCHEDULED_PREFLIGHT_SKIP_REASON",
    "missing_scheduled_capability_reasons",
    "scheduled_certify_capable",
    "scheduled_preflight_capable",
]
