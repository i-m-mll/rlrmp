"""Frozen one-way readers for pre-canonical rendered training payloads.

LEGACY (frozen read boundary, issue b2562ad): these migrations only recognize
fully rendered runtime payloads that predate the embedded ``config`` snapshot.
They are not authoring schemas and must never supply scientific defaults.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def migrate_frozen_rendered_training_payload(
    config_name: str,
    payload: Mapping[str, Any],
    *,
    field_names: frozenset[str],
) -> dict[str, Any] | None:
    """Return a canonical config dict for one recognized rendered payload."""

    canonical = {key: payload[key] for key in field_names.intersection(payload)}
    if config_name == "FixedTargetPerturbationTrainingConfig":
        if not _has_mappings(payload, "sampling", "families"):
            return None
        return canonical
    if config_name == "TargetRelativeMultiTargetTrainingConfig":
        distribution = _mapping(payload.get("target_distribution"))
        if distribution.get("kind") != "structured_static_targets" or not isinstance(
            payload.get("validation_bins"), list | tuple
        ):
            return None
        canonical.update(
            {
                key: distribution[key]
                for key in (
                    "target_support_profile",
                    "seen_directions_deg",
                    "held_out_directions_deg",
                    "seen_amplitudes_m",
                    "held_out_amplitudes_m",
                    "original_target_anchor_m",
                    "support_metadata",
                )
                if key in distribution
            }
        )
        return canonical
    if config_name == "BroadFullStateEpsilonTrainingConfig":
        if not _has_mappings(payload, "sampling", "epsilon_channel", "budget_contract"):
            return None
        return canonical
    if config_name == "PgdFullStateEpsilonTrainingConfig":
        return _migrate_rendered_pgd(payload, canonical)
    if config_name == "PolicyFullStateEpsilonTrainingConfig":
        return _migrate_rendered_policy(payload, canonical)
    return None


def _migrate_rendered_pgd(
    payload: Mapping[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any] | None:
    if not _has_mappings(payload, "inner_maximizer", "epsilon_channel", "budget_contract"):
        return None
    inner = _mapping(payload["inner_maximizer"])
    adam = _mapping(inner.get("adam"))
    schedule = _mapping(payload.get("budget_schedule"))
    objective = _mapping(payload.get("objective"))
    safety_cap = _mapping(payload.get("safety_cap"))
    budget = _mapping(payload.get("budget_contract"))
    budget_source = _mapping(budget.get("budget_source"))
    mechanism = _mapping(payload.get("mechanism"))
    source = _mapping(safety_cap.get("source"))
    max_source = _mapping(schedule.get("max_radius_source"))
    conditioning = _mapping(schedule.get("conditioning_scalar"))
    analytical_sisu_source = _is_analytical_budget_source(max_source)
    _copy_present(
        canonical,
        {
            "adversary_mechanism": _first(mechanism, "name", "policy_class"),
            "n_steps": inner.get("n_steps"),
            "step_size_fraction": _first(
                inner, "step_size_fraction_of_l2_radius", "step_size_fraction"
            ),
            "inner_optimizer_method": inner.get("method"),
            "adam_learning_rate": _first(adam, "learning_rate") or inner.get("learning_rate"),
            "adam_b1": adam.get("b1"),
            "adam_b2": adam.get("b2"),
            "adam_eps": adam.get("eps"),
            "init": _first(inner, "initialization", "init"),
            "budget_schedule": schedule.get("mode"),
            "sisu_levels": schedule.get("levels"),
            "sisu_exact_zero_mass": schedule.get("exact_zero_mass"),
            "sisu_condition_input": _first(schedule, "conditioning_input")
            or conditioning.get("input_key"),
            "sisu_max_l2_radius_15cm": (
                None if analytical_sisu_source else schedule.get("max_l2_radius_15cm")
            ),
            "sisu_max_radius_source": (
                None
                if analytical_sisu_source
                else schedule.get("max_radius_source_key") or max_source.get("key")
            ),
            "objective_kind": objective.get("kind"),
            "energy_gamma_star": objective.get("gamma_star"),
            "energy_gamma_factor": objective.get("gamma_factor"),
            "energy_gamma": objective.get("gamma"),
            "energy_penalty_scale": objective.get("penalty_scale_c"),
            "energy_lambda": objective.get("lambda"),
            "safety_cap_l2_radius_15cm": safety_cap.get("l2_radius_15cm"),
            "safety_cap_source": source.get("key"),
        },
    )
    if schedule.get("mode") == "fixed":
        analytical_fixed_source = _is_analytical_budget_source(budget_source)
        _copy_present(
            canonical,
            {
                "fixed_l2_radius_15cm": (
                    None
                    if analytical_fixed_source
                    else budget.get("effective_l2_radius_15cm")
                ),
                "fixed_radius_source": (
                    None if analytical_fixed_source else budget_source.get("key")
                ),
            },
        )
    return canonical


def _migrate_rendered_policy(
    payload: Mapping[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any] | None:
    if not _has_mappings(payload, "policy", "inner_optimizer", "objective"):
        return None
    policy = _mapping(payload["policy"])
    optimizer = _mapping(payload["inner_optimizer"])
    objective = _mapping(payload["objective"])
    budget = _mapping(payload.get("budget_contract"))
    budget_source = _mapping(budget.get("budget_source"))
    policy_kind = policy.get("kind")
    if policy_kind == "closed_loop_finite_time_varying_epsilon_policy":
        policy_kind = _mapping(policy.get("metadata")).get("policy_class")
    _copy_present(
        canonical,
        {
            "policy_class": policy_kind,
            "mode": objective.get("active"),
            "width": policy.get("width"),
            "depth": policy.get("depth"),
            "n_steps": optimizer.get("n_ascent_steps_per_controller_step"),
            "learning_rate": optimizer.get("learning_rate"),
            "energy_penalty_gamma": objective.get("energy_penalty_gamma"),
            "reference_l2_radius_15cm": budget.get("effective_l2_radius_15cm")
            or budget.get("active_max_l2_radius_15cm"),
            "epsilon_dim": policy.get("output_dim"),
            "state_feature_dim": policy.get("state_feature_dim"),
            "budget_source": budget_source.get("key"),
        },
    )
    return canonical


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _has_mappings(payload: Mapping[str, Any], *keys: str) -> bool:
    return all(bool(_mapping(payload.get(key))) for key in keys)


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _is_analytical_budget_source(source: Mapping[str, Any]) -> bool:
    key = source.get("key")
    return source.get("source_kind") == "analytical_broad_epsilon_anchor" or (
        isinstance(key, str) and key.startswith("analytical_broad_epsilon_level:")
    )


def _copy_present(target: dict[str, Any], values: Mapping[str, Any]) -> None:
    target.update({key: value for key, value in values.items() if value is not None})
