"""Tests for staged recurrent Jacobian diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rlrmp.eval.recurrent_jacobians import (
    CONTEXT,
    CONTROLLER_VISIBLE_FEEDBACK,
    READOUT_STATE_POST_UPDATE,
    SISU,
    STORED_STATE_PRE_UPDATE,
    compute_recurrent_jacobian_blocks,
    compute_recurrent_jacobian_bank,
)


A = jnp.array(
    [
        [0.5, -0.2],
        [0.1, 0.4],
    ],
    dtype=jnp.float32,
)
B_Y = jnp.array(
    [
        [1.0, -1.5],
        [0.25, 0.75],
    ],
    dtype=jnp.float32,
)
B_S = jnp.array(
    [
        [2.0],
        [-0.5],
    ],
    dtype=jnp.float32,
)
B_C = jnp.array(
    [
        [0.3, -0.4],
        [0.2, 0.6],
    ],
    dtype=jnp.float32,
)
BIAS = jnp.array([0.1, -0.2], dtype=jnp.float32)
W = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
READOUT_BIAS = jnp.array([0.3], dtype=jnp.float32)


def _staged_update(h_pre, feedback, sisu, context):
    context_term = 0.0 if context is None else B_C @ jnp.ravel(context)
    return A @ jnp.ravel(h_pre) + B_Y @ jnp.ravel(feedback) + B_S @ jnp.ravel(sisu) + context_term + BIAS


def _readout(h_post):
    return W @ jnp.ravel(h_post) + READOUT_BIAS


def test_recurrent_jacobian_bank_matches_known_hold_context_derivatives() -> None:
    bank = compute_recurrent_jacobian_bank(
        staged_update=_staged_update,
        readout=_readout,
        h_pre=jnp.array([0.2, -0.3], dtype=jnp.float32),
        feedback=jnp.array([1.0, -2.0], dtype=jnp.float32),
        sisu=jnp.array([0.7], dtype=jnp.float32),
        context=jnp.array([0.4, -0.6], dtype=jnp.float32),
        finite_difference=True,
        finite_difference_epsilon=1e-2,
    )

    np.testing.assert_allclose(bank.A, A)
    np.testing.assert_allclose(bank.B_y, B_Y)
    np.testing.assert_allclose(bank.B_s, B_S)
    np.testing.assert_allclose(bank.B_c, B_C)
    np.testing.assert_allclose(bank.W, W)
    np.testing.assert_allclose(bank.K_y, W @ B_Y)
    np.testing.assert_allclose(bank.K_s, W @ B_S)
    np.testing.assert_allclose(bank.K_h, W @ A)
    np.testing.assert_allclose(bank.K_c, W @ B_C)

    metadata = bank.metadata
    assert metadata["domains"][STORED_STATE_PRE_UPDATE]["symbol"] == "h_pre"
    assert metadata["domains"][READOUT_STATE_POST_UPDATE]["symbol"] == "h_post"
    assert metadata["domains"][CONTROLLER_VISIBLE_FEEDBACK]["symbol"] == "y_t"
    assert metadata["domains"][SISU]["symbol"] == "s_t"
    assert metadata["domains"][CONTEXT]["status"] == "available"
    assert metadata["staging"]["output_feedthrough"] == (
        "not_modeled_current_contract_uses_readout_from_h_post"
    )

    matrix_summaries = bank.summaries["matrix_summaries"]
    assert matrix_summaries["A"]["domain"] == STORED_STATE_PRE_UPDATE
    assert matrix_summaries["A"]["codomain"] == READOUT_STATE_POST_UPDATE
    assert matrix_summaries["W"]["domain"] == READOUT_STATE_POST_UPDATE
    assert matrix_summaries["W"]["codomain"] == "action_output"
    assert matrix_summaries["A"]["spectral_radius"] is not None
    assert len(matrix_summaries["A"]["singular_values"]) == 2
    assert matrix_summaries["K_c"]["domain"] == CONTEXT
    assert matrix_summaries["K_c"]["codomain"] == "action_output"

    context_norm = bank.summaries["input_block_norms"][CONTEXT]
    assert context_norm["status"] == "available"
    assert context_norm["fro_norm"] > 0.0

    potent_rows = bank.summaries["output_potent_null_fractions"]["readout_state_maps"]
    assert potent_rows["context_to_readout_state_post_update"]["status"] == "available"
    assert 0.0 <= potent_rows["sisu_to_readout_state_post_update"]["output_potent_fraction"] <= 1.0
    assert 0.0 <= potent_rows["sisu_to_readout_state_post_update"]["output_null_fraction"] <= 1.0

    for block in ("A", "B_y", "B_s", "B_c", "W"):
        check = bank.finite_difference[block]
        assert check["status"] == "available"
        assert check["max_abs_error"] < 5e-5


def test_recurrent_jacobian_bank_marks_context_absent_not_applicable() -> None:
    bank = compute_recurrent_jacobian_bank(
        staged_update=_staged_update,
        readout=_readout,
        h_pre=jnp.array([-0.2, 0.5], dtype=jnp.float32),
        feedback=jnp.array([0.8, 0.1], dtype=jnp.float32),
        sisu=jnp.array([1.2], dtype=jnp.float32),
        context=None,
        finite_difference=True,
        finite_difference_epsilon=1e-2,
    )

    assert bank.B_c is None
    assert bank.K_c is None
    np.testing.assert_allclose(bank.A, A)
    np.testing.assert_allclose(bank.B_y, B_Y)
    np.testing.assert_allclose(bank.B_s, B_S)

    assert bank.metadata["domains"][CONTEXT]["status"] == "not_applicable"
    assert bank.metadata["domains"][CONTEXT]["reason"] == "context_absent"
    assert bank.summaries["context_status"]["hold_only_preparatory_readouts"] == (
        "not_applicable"
    )
    assert bank.summaries["matrix_summaries"]["B_c"]["status"] == "not_applicable"
    assert bank.summaries["matrix_summaries"]["K_c"]["status"] == "not_applicable"
    assert bank.summaries["input_block_norms"][CONTEXT]["status"] == "not_applicable"
    assert (
        bank.summaries["output_potent_null_fractions"]["readout_state_maps"][
            "context_to_readout_state_post_update"
        ]["status"]
        == "not_applicable"
    )
    assert bank.finite_difference["B_c"]["status"] == "not_applicable"

    compact = bank.as_dict()
    assert "arrays" not in compact
    expanded = bank.as_dict(include_arrays=True)
    assert expanded["arrays"]["B_c"] is None
    assert expanded["arrays"]["K_c"] is None
    assert expanded["format"] == "rlrmp.recurrent_jacobian_bank.v1"


def test_recurrent_jacobian_blocks_are_vmap_and_jit_friendly() -> None:
    h_batch = jnp.asarray([[0.2, -0.3], [-0.1, 0.5]], dtype=jnp.float32)
    feedback_batch = jnp.asarray([[1.0, -2.0], [0.8, 0.1]], dtype=jnp.float32)
    sisu = jnp.asarray([0.7], dtype=jnp.float32)
    context = jnp.asarray([0.4, -0.6], dtype=jnp.float32)

    def one_row(h_pre, feedback):
        return compute_recurrent_jacobian_blocks(
            staged_update=_staged_update,
            readout=_readout,
            h_pre=h_pre,
            feedback=feedback,
            sisu=sisu,
            context=context,
        )

    batched = jax.vmap(one_row)(h_batch, feedback_batch)
    jitted = jax.jit(lambda h: one_row(h, feedback_batch[0]).K_c)(h_batch[0])

    np.testing.assert_allclose(batched.A, jnp.broadcast_to(A, (2,) + A.shape))
    np.testing.assert_allclose(batched.K_c, jnp.broadcast_to(W @ B_C, (2, 1, 2)))
    np.testing.assert_allclose(jitted, W @ B_C)
