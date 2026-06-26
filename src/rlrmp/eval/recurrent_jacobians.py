"""Staged recurrent-controller Jacobian diagnostics.

This module works at the current rlrmp staged recurrent convention:

``h_post = staged_update(h_pre, y, s, c)``
``u = readout(h_post)``

where ``h_pre`` is the stored hidden state before the recurrent update,
``h_post`` is the post-update/readout state, ``y`` is controller-visible
feedback, ``s`` is the SISU signal, and ``c`` is an optional non-SISU context
block such as a hold-task go-cue scalar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

STORED_STATE_PRE_UPDATE = "stored_state_pre_update"
READOUT_STATE_POST_UPDATE = "readout_state_post_update"
CONTROLLER_VISIBLE_FEEDBACK = "controller_visible_feedback"
SISU = "sisu"
CONTEXT = "context"
ACTION_OUTPUT = "action_output"

BLOCK_LABELS = {
    "A": "dh_post_dh_pre",
    "B_y": "dh_post_dy",
    "B_s": "dh_post_ds",
    "B_c": "dh_post_dc",
    "W": "du_dh_post",
    "K_y": "du_dy",
    "K_s": "du_ds",
    "K_h": "du_dh_pre",
    "K_c": "du_dc",
}

DOMAIN_SCHEMA = {
    STORED_STATE_PRE_UPDATE: {
        "symbol": "h_pre",
        "timing": "stored before recurrent update",
    },
    READOUT_STATE_POST_UPDATE: {
        "symbol": "h_post",
        "timing": "post-update state used by the action readout",
    },
    CONTROLLER_VISIBLE_FEEDBACK: {
        "symbol": "y_t",
        "timing": "current feedback visible to the controller update",
    },
    SISU: {
        "symbol": "s_t",
        "timing": "current SISU signal visible to the controller update",
    },
    CONTEXT: {
        "symbol": "c_t",
        "timing": "optional current non-SISU context visible to the update",
    },
    ACTION_OUTPUT: {
        "symbol": "u_t",
        "timing": "linear or local readout from h_post",
    },
}


class RecurrentJacobianBlocks(NamedTuple):
    """Pure array recurrent Jacobian blocks for JAX transforms."""

    A: jax.Array
    B_y: jax.Array
    B_s: jax.Array
    B_c: jax.Array | None
    W: jax.Array
    K_y: jax.Array
    K_s: jax.Array
    K_h: jax.Array
    K_c: jax.Array | None
    h_post: jax.Array
    u: jax.Array


@dataclass(frozen=True)
class RecurrentJacobianBank:
    """Local staged recurrent derivative bank.

    Arrays use flattened vector bases for each local block. The primary hidden
    blocks follow the staged convention:

    - ``A`` has shape ``(hidden, hidden)`` and equals ``dh_post / dh_pre``.
    - ``B_y`` has shape ``(hidden, feedback)``.
    - ``B_s`` has shape ``(hidden, sisu)``.
    - ``B_c`` has shape ``(hidden, context)`` when context is present, else
      ``None``.
    - ``W`` has shape ``(action, hidden)`` and equals ``du / dh_post``.
    - ``K_y``, ``K_s``, ``K_h``, and optional ``K_c`` are readout-composed
      output maps.
    """

    A: jax.Array
    B_y: jax.Array
    B_s: jax.Array
    B_c: jax.Array | None
    W: jax.Array
    K_y: jax.Array
    K_s: jax.Array
    K_h: jax.Array
    K_c: jax.Array | None
    h_post: jax.Array
    u: jax.Array
    metadata: dict[str, Any]
    summaries: dict[str, Any]
    finite_difference: dict[str, Any]

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a serialization-friendly representation.

        Args:
            include_arrays: Include matrix and point values as nested Python
                lists. Leave false for compact metadata-only reports.

        Returns:
            Dictionary containing metadata, summaries, and optional arrays.
        """

        result: dict[str, Any] = {
            "format": "rlrmp.recurrent_jacobian_bank.v1",
            "metadata": self.metadata,
            "summaries": self.summaries,
            "finite_difference": self.finite_difference,
        }
        if include_arrays:
            result["arrays"] = {
                "A": _array_to_list(self.A),
                "B_y": _array_to_list(self.B_y),
                "B_s": _array_to_list(self.B_s),
                "B_c": None if self.B_c is None else _array_to_list(self.B_c),
                "W": _array_to_list(self.W),
                "K_y": _array_to_list(self.K_y),
                "K_s": _array_to_list(self.K_s),
                "K_h": _array_to_list(self.K_h),
                "K_c": None if self.K_c is None else _array_to_list(self.K_c),
                "h_post": _array_to_list(self.h_post),
                "u": _array_to_list(self.u),
            }
        return result


def compute_recurrent_jacobian_blocks(
    *,
    staged_update: Callable[[jax.Array, jax.Array, jax.Array, jax.Array | None], jax.Array],
    readout: Callable[[jax.Array], jax.Array],
    h_pre: jax.Array,
    feedback: jax.Array,
    sisu: jax.Array,
    context: jax.Array | None = None,
) -> RecurrentJacobianBlocks:
    """Compute exact staged recurrent derivative blocks without host summaries.

    This function returns only JAX arrays and ``None`` for absent context blocks,
    so it can be used inside ``jax.vmap``, ``jax.jit``, or ``jax.lax.scan`` over
    sampled states or trajectory rows. Serialization, SVD/eigendecompositions,
    and finite-difference checks are intentionally kept in
    :func:`compute_recurrent_jacobian_bank`.
    """

    h_pre = jnp.asarray(h_pre)
    feedback = jnp.asarray(feedback)
    sisu = jnp.asarray(sisu)
    context = None if context is None else jnp.asarray(context)

    h_flat = jnp.ravel(h_pre)
    feedback_flat = jnp.ravel(feedback)
    sisu_flat = jnp.ravel(sisu)

    if context is None:

        def update_from_flats(
            flat_h: jax.Array,
            flat_feedback: jax.Array,
            flat_sisu: jax.Array,
        ) -> jax.Array:
            return jnp.ravel(
                staged_update(
                    flat_h.reshape(h_pre.shape),
                    flat_feedback.reshape(feedback.shape),
                    flat_sisu.reshape(sisu.shape),
                    None,
                )
            )

        h_post = update_from_flats(h_flat, feedback_flat, sisu_flat)
        A, B_y, B_s = jax.jacfwd(update_from_flats, argnums=(0, 1, 2))(
            h_flat,
            feedback_flat,
            sisu_flat,
        )
        B_c = None
    else:
        context_flat = jnp.ravel(context)

        def update_from_flats(
            flat_h: jax.Array,
            flat_feedback: jax.Array,
            flat_sisu: jax.Array,
            flat_context: jax.Array,
        ) -> jax.Array:
            return jnp.ravel(
                staged_update(
                    flat_h.reshape(h_pre.shape),
                    flat_feedback.reshape(feedback.shape),
                    flat_sisu.reshape(sisu.shape),
                    flat_context.reshape(context.shape),
                )
            )

        h_post = update_from_flats(h_flat, feedback_flat, sisu_flat, context_flat)
        A, B_y, B_s, B_c = jax.jacfwd(update_from_flats, argnums=(0, 1, 2, 3))(
            h_flat,
            feedback_flat,
            sisu_flat,
            context_flat,
        )

    u = jnp.ravel(readout(h_post))
    W = _jacobian_matrix(readout, h_post)
    K_y = W @ B_y
    K_s = W @ B_s
    K_h = W @ A
    K_c = None if B_c is None else W @ B_c
    return RecurrentJacobianBlocks(
        A=A,
        B_y=B_y,
        B_s=B_s,
        B_c=B_c,
        W=W,
        K_y=K_y,
        K_s=K_s,
        K_h=K_h,
        K_c=K_c,
        h_post=h_post,
        u=u,
    )


def compute_recurrent_jacobian_bank(
    *,
    staged_update: Callable[[jax.Array, jax.Array, jax.Array, jax.Array | None], jax.Array],
    readout: Callable[[jax.Array], jax.Array],
    h_pre: jax.Array,
    feedback: jax.Array,
    sisu: jax.Array,
    context: jax.Array | None = None,
    finite_difference: bool = False,
    finite_difference_epsilon: float = 1e-4,
    finite_difference_batch_size: int | None = 128,
) -> RecurrentJacobianBank:
    """Compute local staged recurrent Jacobian diagnostics.

    Args:
        staged_update: Callable implementing
            ``h_post = staged_update(h_pre, feedback, sisu, context)``.
        readout: Callable implementing ``u = readout(h_post)``.
        h_pre: Stored hidden state before the recurrent update.
        feedback: Current controller-visible feedback/input ``y_t``.
        sisu: Current SISU block ``s_t``. Scalars are treated as one-channel
            blocks.
        context: Optional non-SISU context block ``c_t``. When absent, context
            derivative summaries are marked ``not_applicable``.
        finite_difference: Whether to include central finite-difference sanity
            checks for each local block. Disabled by default because the exact
            array kernel is the normal mapped/JIT path.
        finite_difference_epsilon: Step size for finite-difference checks.
        finite_difference_batch_size: Batch size for finite-difference
            ``lax.map`` coordinate sweeps.

    Returns:
        A :class:`RecurrentJacobianBank` with exact autodiff blocks, derived
        readout-composed maps, explicit timing/domain metadata, summaries, and
        finite-difference checks.
    """

    blocks = compute_recurrent_jacobian_blocks(
        staged_update=staged_update,
        readout=readout,
        h_pre=h_pre,
        feedback=feedback,
        sisu=sisu,
        context=context,
    )
    h_pre = jnp.asarray(h_pre)
    feedback = jnp.asarray(feedback)
    sisu = jnp.asarray(sisu)
    context = None if context is None else jnp.asarray(context)

    matrix_blocks = {
        "A": blocks.A,
        "B_y": blocks.B_y,
        "B_s": blocks.B_s,
        "W": blocks.W,
        "K_y": blocks.K_y,
        "K_s": blocks.K_s,
        "K_h": blocks.K_h,
    }
    if blocks.B_c is not None:
        matrix_blocks["B_c"] = blocks.B_c
    if blocks.K_c is not None:
        matrix_blocks["K_c"] = blocks.K_c

    summaries = {
        "matrix_summaries": _matrix_summaries(
            matrix_blocks,
            context_present=context is not None,
        ),
        "input_block_norms": _input_block_norms(
            B_y=blocks.B_y,
            B_s=blocks.B_s,
            B_c=blocks.B_c,
        ),
        "output_potent_null_fractions": _output_potent_null_fractions(
            W=blocks.W,
            maps={
                "stored_state_pre_update_to_readout_state_post_update": blocks.A,
                "controller_visible_feedback_to_readout_state_post_update": blocks.B_y,
                "sisu_to_readout_state_post_update": blocks.B_s,
                **(
                    {"context_to_readout_state_post_update": blocks.B_c}
                    if blocks.B_c is not None
                    else {}
                ),
            },
            context_present=context is not None,
        ),
        "context_status": _context_status(context),
    }

    fd_checks = (
        finite_difference_sanity_checks(
            staged_update=staged_update,
            readout=readout,
            h_pre=h_pre,
            feedback=feedback,
            sisu=sisu,
            context=context,
            bank_blocks={
                "A": blocks.A,
                "B_y": blocks.B_y,
                "B_s": blocks.B_s,
                "B_c": blocks.B_c,
                "W": blocks.W,
            },
            epsilon=finite_difference_epsilon,
            batch_size=finite_difference_batch_size,
        )
        if finite_difference
        else {"status": "skipped"}
    )

    return RecurrentJacobianBank(
        A=blocks.A,
        B_y=blocks.B_y,
        B_s=blocks.B_s,
        B_c=blocks.B_c,
        W=blocks.W,
        K_y=blocks.K_y,
        K_s=blocks.K_s,
        K_h=blocks.K_h,
        K_c=blocks.K_c,
        h_post=blocks.h_post,
        u=blocks.u,
        metadata=_metadata(
            h_pre=h_pre,
            feedback=feedback,
            sisu=sisu,
            context=context,
            h_post=blocks.h_post,
            u=blocks.u,
        ),
        summaries=summaries,
        finite_difference=fd_checks,
    )


def finite_difference_sanity_checks(
    *,
    staged_update: Callable[[jax.Array, jax.Array, jax.Array, jax.Array | None], jax.Array],
    readout: Callable[[jax.Array], jax.Array],
    h_pre: jax.Array,
    feedback: jax.Array,
    sisu: jax.Array,
    context: jax.Array | None,
    bank_blocks: Mapping[str, jax.Array | None],
    epsilon: float = 1e-4,
    batch_size: int | None = 128,
) -> dict[str, Any]:
    """Return finite-difference sanity checks for local derivative blocks."""

    h_post = jnp.ravel(staged_update(h_pre, feedback, sisu, context))
    checks: dict[str, Any] = {
        "A": _finite_difference_check(
            lambda h: staged_update(h, feedback, sisu, context),
            h_pre,
            bank_blocks["A"],
            epsilon=epsilon,
            batch_size=batch_size,
        ),
        "B_y": _finite_difference_check(
            lambda y: staged_update(h_pre, y, sisu, context),
            feedback,
            bank_blocks["B_y"],
            epsilon=epsilon,
            batch_size=batch_size,
        ),
        "B_s": _finite_difference_check(
            lambda s: staged_update(h_pre, feedback, s, context),
            sisu,
            bank_blocks["B_s"],
            epsilon=epsilon,
            batch_size=batch_size,
        ),
        "W": _finite_difference_check(
            readout,
            h_post,
            bank_blocks["W"],
            epsilon=epsilon,
            batch_size=batch_size,
        ),
    }
    if context is None:
        checks["B_c"] = {
            "status": "not_applicable",
            "reason": "context_absent",
            "epsilon": float(epsilon),
        }
    else:
        checks["B_c"] = _finite_difference_check(
            lambda c: staged_update(h_pre, feedback, sisu, c),
            context,
            bank_blocks["B_c"],
            epsilon=epsilon,
            batch_size=batch_size,
        )
    return checks


def _metadata(
    *,
    h_pre: jax.Array,
    feedback: jax.Array,
    sisu: jax.Array,
    context: jax.Array | None,
    h_post: jax.Array,
    u: jax.Array,
) -> dict[str, Any]:
    domains = dict(DOMAIN_SCHEMA)
    domains[CONTEXT] = {
        **domains[CONTEXT],
        "status": "available" if context is not None else "not_applicable",
        "reason": None if context is not None else "context_absent",
    }
    return {
        "staging": {
            "stored_state_pre_update": "h_pre is sampled before the recurrent update",
            "readout_state_post_update": "h_post is computed before the readout",
            "output_feedthrough": "not_modeled_current_contract_uses_readout_from_h_post",
        },
        "domains": domains,
        "shapes": {
            STORED_STATE_PRE_UPDATE: tuple(int(dim) for dim in h_pre.shape),
            CONTROLLER_VISIBLE_FEEDBACK: tuple(int(dim) for dim in feedback.shape),
            SISU: tuple(int(dim) for dim in sisu.shape),
            CONTEXT: None if context is None else tuple(int(dim) for dim in context.shape),
            READOUT_STATE_POST_UPDATE: tuple(int(dim) for dim in h_post.shape),
            ACTION_OUTPUT: tuple(int(dim) for dim in u.shape),
        },
        "block_labels": BLOCK_LABELS,
    }


def _matrix_summaries(
    blocks: Mapping[str, jax.Array],
    *,
    context_present: bool,
) -> dict[str, Any]:
    summaries = {
        name: _matrix_summary(
            name,
            matrix,
            domain=_block_domain(name),
            codomain=_block_codomain(name),
        )
        for name, matrix in blocks.items()
    }
    if not context_present:
        summaries["B_c"] = {
            "status": "not_applicable",
            "reason": "context_absent",
            "domain": CONTEXT,
            "codomain": READOUT_STATE_POST_UPDATE,
        }
        summaries["K_c"] = {
            "status": "not_applicable",
            "reason": "context_absent",
            "domain": CONTEXT,
            "codomain": ACTION_OUTPUT,
        }
    return summaries


def _matrix_summary(
    name: str,
    matrix: jax.Array,
    *,
    domain: str,
    codomain: str,
    rank_tol: float = 1e-10,
) -> dict[str, Any]:
    arr = _to_numpy(matrix)
    singular_values = np.linalg.svd(arr, compute_uv=False)
    summary: dict[str, Any] = {
        "status": "available",
        "name": name,
        "label": BLOCK_LABELS[name],
        "domain": domain,
        "codomain": codomain,
        "shape": tuple(int(dim) for dim in arr.shape),
        "fro_norm": float(np.linalg.norm(arr, ord="fro")),
        "spectral_norm": float(singular_values[0]) if singular_values.size else 0.0,
        "singular_values": [float(value) for value in singular_values],
        "rank": int(np.sum(singular_values > rank_tol)),
    }
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.size:
        eigenvalues = np.linalg.eigvals(arr)
        summary["spectral_radius"] = float(np.max(np.abs(eigenvalues)))
        summary["eigenvalues"] = [
            {
                "real": float(np.real(value)),
                "imag": float(np.imag(value)),
                "abs": float(np.abs(value)),
            }
            for value in eigenvalues
        ]
    else:
        summary["spectral_radius"] = None
        summary["eigenvalues"] = []
    return summary


def _input_block_norms(
    *,
    B_y: jax.Array,
    B_s: jax.Array,
    B_c: jax.Array | None,
) -> dict[str, Any]:
    norms = {
        CONTROLLER_VISIBLE_FEEDBACK: _norm_summary(B_y),
        SISU: _norm_summary(B_s),
    }
    norms[CONTEXT] = (
        _norm_summary(B_c)
        if B_c is not None
        else {"status": "not_applicable", "reason": "context_absent"}
    )
    return norms


def _norm_summary(matrix: jax.Array) -> dict[str, Any]:
    arr = _to_numpy(matrix)
    column_norms = np.linalg.norm(arr, axis=0) if arr.size else np.zeros((arr.shape[1],))
    return {
        "status": "available",
        "fro_norm": float(np.linalg.norm(arr, ord="fro")),
        "column_norms": [float(value) for value in column_norms],
    }


def _output_potent_null_fractions(
    *,
    W: jax.Array,
    maps: Mapping[str, jax.Array],
    context_present: bool,
    rank_tol: float = 1e-10,
) -> dict[str, Any]:
    readout = _to_numpy(W)
    if readout.size == 0:
        basis = np.zeros((0, readout.shape[1]), dtype=np.float64)
    else:
        _u, singular_values, vh = np.linalg.svd(readout, full_matrices=False)
        rank = int(np.sum(singular_values > rank_tol))
        basis = vh[:rank, :]

    rows = {
        label: _potent_null_summary(matrix, basis)
        for label, matrix in maps.items()
    }
    if not context_present:
        rows["context_to_readout_state_post_update"] = {
            "status": "not_applicable",
            "reason": "context_absent",
        }
    return {
        "readout_rowspace_rank": int(basis.shape[0]),
        "readout_state_maps": rows,
    }


def _potent_null_summary(matrix: jax.Array, basis: np.ndarray) -> dict[str, Any]:
    arr = _to_numpy(matrix)
    if basis.size == 0:
        potent = np.zeros_like(arr)
    else:
        potent = basis.T @ (basis @ arr)
    null = arr - potent
    total_energy = float(np.sum(arr * arr))
    potent_energy = float(np.sum(potent * potent))
    null_energy = float(np.sum(null * null))
    if total_energy <= 0.0:
        potent_fraction = 0.0
        null_fraction = 0.0
    else:
        potent_fraction = potent_energy / total_energy
        null_fraction = null_energy / total_energy
    return {
        "status": "available",
        "total_energy": total_energy,
        "output_potent_energy": potent_energy,
        "output_null_energy": null_energy,
        "output_potent_fraction": potent_fraction,
        "output_null_fraction": null_fraction,
    }


def _context_status(context: jax.Array | None) -> dict[str, Any]:
    if context is None:
        return {
            "status": "not_applicable",
            "reason": "context_absent",
            "hold_only_preparatory_readouts": "not_applicable",
        }
    return {
        "status": "available",
        "reason": None,
        "hold_only_preparatory_readouts": "available_if_supplied_by_downstream_analysis",
    }


def _finite_difference_check(
    func: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    expected: jax.Array | None,
    *,
    epsilon: float,
    batch_size: int | None,
) -> dict[str, Any]:
    if expected is None:
        return {
            "status": "not_applicable",
            "reason": "block_absent",
            "epsilon": float(epsilon),
        }
    fd = _finite_difference_matrix(func, x, epsilon=epsilon, batch_size=batch_size)
    expected_arr = _to_numpy(expected)
    error = fd - expected_arr
    expected_norm = float(np.linalg.norm(expected_arr, ord="fro"))
    return {
        "status": "available",
        "epsilon": float(epsilon),
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
        "fro_error": float(np.linalg.norm(error, ord="fro")),
        "relative_fro_error": (
            float(np.linalg.norm(error, ord="fro") / expected_norm)
            if expected_norm > 0.0
            else float(np.linalg.norm(error, ord="fro"))
        ),
    }


def _finite_difference_matrix(
    func: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    epsilon: float,
    batch_size: int | None,
) -> np.ndarray:
    x = jnp.asarray(x)
    x_flat = jnp.ravel(x)
    if x_flat.size == 0:
        output_size = int(jnp.ravel(func(x)).size)
        return np.zeros((output_size, 0), dtype=np.float64)

    def column(index: jax.Array) -> jax.Array:
        direction = jax.nn.one_hot(index, x_flat.size, dtype=x_flat.dtype)
        step = epsilon * direction
        plus = jnp.reshape(x_flat + step, x.shape)
        minus = jnp.reshape(x_flat - step, x.shape)
        return (jnp.ravel(func(plus)) - jnp.ravel(func(minus))) / (2.0 * epsilon)

    indices = jnp.arange(x_flat.size, dtype=jnp.int32)
    return _to_numpy(jax.lax.map(column, indices, batch_size=batch_size).T)


def _jacobian_matrix(
    func: Callable[[jax.Array], jax.Array],
    x: jax.Array,
) -> jax.Array:
    x = jnp.asarray(x)
    x_flat = jnp.ravel(x)

    def flat_func(flat_x: jax.Array) -> jax.Array:
        return jnp.ravel(func(jnp.reshape(flat_x, x.shape)))

    return jax.jacfwd(flat_func)(x_flat)


def _block_domain(name: str) -> str:
    return {
        "A": STORED_STATE_PRE_UPDATE,
        "B_y": CONTROLLER_VISIBLE_FEEDBACK,
        "B_s": SISU,
        "B_c": CONTEXT,
        "W": READOUT_STATE_POST_UPDATE,
        "K_y": CONTROLLER_VISIBLE_FEEDBACK,
        "K_s": SISU,
        "K_h": STORED_STATE_PRE_UPDATE,
        "K_c": CONTEXT,
    }[name]


def _block_codomain(name: str) -> str:
    return READOUT_STATE_POST_UPDATE if name in {"A", "B_y", "B_s", "B_c"} else ACTION_OUTPUT


def _to_numpy(value: jax.Array) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _array_to_list(value: jax.Array) -> list[Any]:
    return np.asarray(jax.device_get(value)).tolist()


__all__ = [
    "ACTION_OUTPUT",
    "BLOCK_LABELS",
    "CONTEXT",
    "CONTROLLER_VISIBLE_FEEDBACK",
    "DOMAIN_SCHEMA",
    "READOUT_STATE_POST_UPDATE",
    "SISU",
    "STORED_STATE_PRE_UPDATE",
    "RecurrentJacobianBlocks",
    "RecurrentJacobianBank",
    "compute_recurrent_jacobian_blocks",
    "compute_recurrent_jacobian_bank",
    "finite_difference_sanity_checks",
]
