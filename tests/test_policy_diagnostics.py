"""Tests for black-box controller-local policy diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from rlrmp.eval.policy_diagnostics import (
    NOT_APPLICABLE,
    PolicyAbsentInputBlock,
    PolicyInputSchema,
    directional_gain_summary,
    feedback_jacobian_sisu_modulation,
    finite_difference_jacobian,
    policy_block_jacobian,
    policy_jacobian,
    signed_pair_odd_even_summary,
    singular_value_summary,
    validate_policy_jacobian,
)


def test_no_hold_schema_marks_context_not_applicable() -> None:
    values = {
        "feedback": jnp.zeros((2,), dtype=jnp.float32),
        "sisu": jnp.zeros((1,), dtype=jnp.float32),
    }
    schema = PolicyInputSchema.from_values(
        values,
        roles={"feedback": "sensory_feedback", "sisu": "sisu_condition"},
        absent_blocks=(
            PolicyAbsentInputBlock(
                name="context",
                role="go_cue_or_hold_context",
                reason="no-hold task family omits the context channel",
            ),
        ),
    )

    payload = schema.to_json()

    assert schema.block_names == ("feedback", "sisu")
    assert schema.block_slice("feedback") == slice(0, 2)
    assert schema.block_slice("sisu") == slice(2, 3)
    assert payload["blocks"][0]["role"] == "sensory_feedback"
    assert payload["absent_blocks"] == [
        {
            "name": "context",
            "role": "go_cue_or_hold_context",
            "status": NOT_APPLICABLE,
            "reason": "no-hold task family omits the context channel",
        }
    ]


def test_hold_schema_includes_named_context_block() -> None:
    values = {
        "feedback": jnp.zeros((2,), dtype=jnp.float32),
        "sisu": jnp.zeros((1,), dtype=jnp.float32),
        "context": jnp.zeros((1,), dtype=jnp.float32),
    }
    schema = PolicyInputSchema.from_values(
        values,
        roles={
            "feedback": "sensory_feedback",
            "sisu": "sisu_condition",
            "context": "go_cue_or_hold_context",
        },
    )

    payload = schema.to_json()

    assert schema.block_names == ("feedback", "sisu", "context")
    assert schema.block_slice("context") == slice(3, 4)
    assert payload["absent_blocks"] == []
    assert payload["blocks"][2]["shape"] == [1]
    assert payload["blocks"][2]["status"] == "available"


def test_policy_jacobian_and_finite_difference_validation() -> None:
    weights_y = jnp.asarray([[2.0, -1.0], [0.5, 3.0]], dtype=jnp.float32)
    weights_s = jnp.asarray([[4.0], [-2.0]], dtype=jnp.float32)
    values = {
        "feedback": jnp.asarray([0.2, -0.3], dtype=jnp.float32),
        "sisu": jnp.asarray([0.7], dtype=jnp.float32),
    }
    schema = PolicyInputSchema.from_values(values)

    def policy(blocks):
        return weights_y @ blocks["feedback"] + weights_s @ blocks["sisu"]

    jac = policy_jacobian(policy, values, schema=schema)
    validation = validate_policy_jacobian(policy, values, schema=schema, epsilon=1e-2)

    np.testing.assert_allclose(np.asarray(jac.block("feedback")), np.asarray(weights_y))
    np.testing.assert_allclose(np.asarray(jac.block("sisu")), np.asarray(weights_s))
    assert jac.full.shape == (2, 3)
    assert validation.passed
    assert validation.max_abs_error < 2e-5


def test_policy_block_jacobian_restricts_to_named_input_block() -> None:
    values = {
        "feedback": jnp.asarray([0.2, -0.3], dtype=jnp.float32),
        "sisu": jnp.asarray([0.7], dtype=jnp.float32),
        "context": jnp.asarray([1.0], dtype=jnp.float32),
    }
    schema = PolicyInputSchema.from_values(values)

    def policy(blocks):
        return jnp.asarray(
            [
                2.0 * blocks["feedback"][0] + 5.0 * blocks["sisu"][0],
                -3.0 * blocks["feedback"][1] + 7.0 * blocks["context"][0],
            ],
            dtype=jnp.float32,
        )

    feedback_jacobian = policy_block_jacobian(
        policy,
        values,
        "feedback",
        schema=schema,
    )
    finite_difference = finite_difference_jacobian(
        policy,
        values,
        schema=schema,
        epsilon=1e-2,
        batch_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(feedback_jacobian),
        np.asarray([[2.0, 0.0], [0.0, -3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(finite_difference[:, schema.block_slice("feedback")]),
        np.asarray(feedback_jacobian),
        rtol=1e-5,
        atol=1e-5,
    )


def test_singular_values_and_directional_gains() -> None:
    matrix = jnp.asarray([[3.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)

    sv = singular_value_summary(matrix)
    directional = directional_gain_summary(
        matrix,
        jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=jnp.float32),
    )

    assert sv["status"] == "available"
    assert sv["singular_values"] == pytest.approx([4.0, 3.0])
    assert sv["spectral_norm"] == pytest.approx(4.0)
    assert directional["status"] == "available"
    assert directional["gains"][:2] == pytest.approx([3.0, 4.0])
    assert directional["gains"][2] == pytest.approx(np.sqrt(25.0 / 2.0))
    assert directional["max_gain"] == pytest.approx(4.0)


def test_sisu_modulation_summarizes_feedback_jacobian_changes() -> None:
    values = {
        "feedback": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "sisu": jnp.asarray([0.0], dtype=jnp.float32),
    }
    schema = PolicyInputSchema.from_values(values)

    def policy(blocks):
        sisu = blocks["sisu"][0]
        gain = jnp.asarray([[1.0 + sisu, 0.0], [0.0, 2.0 - sisu]], dtype=jnp.float32)
        return gain @ blocks["feedback"]

    modulation = feedback_jacobian_sisu_modulation(
        policy,
        values,
        schema=schema,
        sisu_values=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
    )

    assert modulation["status"] == "available"
    np.testing.assert_allclose(
        np.asarray(modulation["jacobians"][0]),
        np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(modulation["endpoint_slope"]),
        np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
    )


def test_signed_pair_odd_even_residuals_allow_baseline() -> None:
    baseline = jnp.asarray([10.0, -2.0], dtype=jnp.float32)
    positive = baseline + jnp.asarray([2.0, 0.5], dtype=jnp.float32)
    negative = baseline + jnp.asarray([-1.0, 0.5], dtype=jnp.float32)

    summary = signed_pair_odd_even_summary(
        positive,
        negative,
        baseline=baseline,
        amplitude=0.25,
    )

    np.testing.assert_allclose(np.asarray(summary["odd_response"]), np.asarray([1.5, 0.0]))
    np.testing.assert_allclose(
        np.asarray(summary["even_nonlinear_residual"]),
        np.asarray([0.5, 0.5]),
    )
    assert summary["odd_norm"] == pytest.approx(1.5)
    assert summary["even_nonlinear_residual_norm"] == pytest.approx(np.sqrt(0.5))
    assert summary["curvature_like_even_norm"] == pytest.approx(np.sqrt(0.5) / 0.25**2)
