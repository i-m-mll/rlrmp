"""Guard canonical run specs against retired flat training-config payload keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlrmp.train.training_configs import (
    FixedTargetPerturbationTrainingConfig,
    PgdFullStateEpsilonTrainingConfig,
)


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SPEC_GLOB = "results/**/runs/*.json"

RETIRED_KEYS_BY_PAYLOAD = {
    "target_relative_multitarget": frozenset(
        {
            "target_support_profile",
            "seen_directions_deg",
            "held_out_directions_deg",
            "seen_amplitudes_m",
            "held_out_amplitudes_m",
            "original_target_anchor_m",
            "support_metadata",
        }
    ),
    "broad_epsilon_pgd_training": frozenset(
        {
            "budget_schedule_mode",
            "fixed_l2_radius_15cm",
            "fixed_radius_source",
            "n_steps",
            "step_size_fraction_of_l2_radius",
            "step_size_fraction",
            "inner_optimizer_method",
            "adam_learning_rate",
            "adam_b1",
            "adam_b2",
            "adam_eps",
            "initialization",
            "init",
            "sisu_levels",
            "sisu_exact_zero_mass",
            "sisu_condition_input",
            "sisu_max_l2_radius_15cm",
            "sisu_max_radius_source",
            "objective_kind",
            "energy_gamma_star",
            "energy_gamma_factor",
            "energy_gamma",
            "energy_penalty_scale",
            "energy_lambda",
            "safety_cap_l2_radius_15cm",
            "safety_cap_source",
        }
    ),
    "policy_adversary_training": frozenset(
        {
            "width",
            "depth",
            "n_steps",
            "learning_rate",
            "energy_penalty_gamma",
            "reference_l2_radius_15cm",
            "state_feature_dim",
            "budget_source",
        }
    ),
}


def test_fixed_target_from_payload_accepts_dicts() -> None:
    config = FixedTargetPerturbationTrainingConfig.from_payload(
        {
            "enabled": True,
            "nominal_fraction": 0.4,
            "single_fraction": 0.5,
            "combined_fraction": 0.1,
            "pulse_start_step": 17,
        }
    )

    assert config.enabled is True
    assert config.nominal_fraction == 0.4
    assert config.single_fraction == 0.5
    assert config.pulse_start_step == 17


def test_from_payload_accepts_canonical_snapshot_and_rejects_legacy_nested_shape() -> None:
    authored = PgdFullStateEpsilonTrainingConfig(
        enabled=True,
        reach_length_scaling=False,
        n_steps=9,
    )

    parsed = PgdFullStateEpsilonTrainingConfig.from_payload(authored.to_hps_dict())

    assert parsed == authored
    with pytest.raises(ValueError, match="[Ee]xtra inputs are not permitted"):
        PgdFullStateEpsilonTrainingConfig.from_payload(
            {
                "enabled": True,
                "inner_maximizer": {
                    "n_steps": 9,
                    "step_size_fraction_of_l2_radius": 0.25,
                },
            }
        )


def test_run_specs_do_not_use_retired_flat_training_config_keys() -> None:
    violations: list[str] = []
    for path in sorted(REPO_ROOT.glob(RUN_SPEC_GLOB)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for location, slot_name, slot_payload in _training_payloads(payload):
            retired = RETIRED_KEYS_BY_PAYLOAD[slot_name].intersection(slot_payload)
            if retired:
                relative_path = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{relative_path}:{location}: {sorted(retired)}")

    assert not violations, (
        "Canonical run specs contain retired flat training-config keys; emit the nested "
        "audit-contract shape instead:\n" + "\n".join(violations)
    )


def _training_payloads(
    value: Any,
    location: str = "$",
) -> list[tuple[str, str, dict[str, Any]]]:
    found: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in RETIRED_KEYS_BY_PAYLOAD and isinstance(child, dict):
                found.append((child_location, key, child))
            found.extend(_training_payloads(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_training_payloads(child, f"{location}[{index}]"))
    return found
