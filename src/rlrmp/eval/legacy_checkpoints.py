"""LEGACY (frozen 2026-07-11, issue ef8e1df) checkpoint readers.

This archival boundary reconstructs the Equinox checkpoint family used by the
020a65b and e901a20 result materializers. It is retained only so those frozen
artifacts remain readable; new checkpoints must use the current Feedbax model
artifact contracts. A migrate-once conversion is deliberately deferred.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.models.networks import MaskedLinear
from jax_cookbook import load_with_hyperparameters

from rlrmp.train.task_model import setup_task_model_pair


NUMERIC_SCALAR_TYPES = (bool, int, float, np.bool_, np.integer, np.floating)
ModelFactory = Callable[[Any, Any], Any]


def _default_model_factory(hps: Any, key: Any) -> Any:
    return setup_task_model_pair(hps, key=key).model


def force_legacy_masked_readout(model: Any) -> Any:
    """Return a template with the historical dynamic readout-mask leaf."""

    net = model.nodes["net"].net
    if getattr(net, "dtype", None) is not jnp.float64:
        # Static dtype is not a PyTree leaf. This is a fresh load template.
        object.__setattr__(net, "dtype", jnp.float64)
        net = model.nodes["net"].net
    if not isinstance(net.readout, eqx.nn.Linear):
        return model
    mask = jnp.ones_like(net.readout.weight, dtype=bool)
    masked_readout = MaskedLinear(
        net.readout.weight.shape[-1],
        net.readout.weight.shape[-2],
        mask,
        use_bias=net.readout.bias is not None,
        dtype=net.readout.weight.dtype,
        key=jr.PRNGKey(0),
    )
    masked_readout = eqx.tree_at(
        lambda layer: (layer.linear.weight, layer.linear.bias),
        masked_readout,
        (net.readout.weight, net.readout.bias),
    )
    return eqx.tree_at(lambda tree: tree.nodes["net"].net.readout, model, masked_readout)


def legacy_static_numeric_template(tree: Any, n_replicates: int) -> Any:
    """Return a template that consumes historical replicated scalar leaves."""

    def replace_numeric_scalar(leaf: Any) -> Any:
        if isinstance(leaf, NUMERIC_SCALAR_TYPES):
            return jnp.full((n_replicates,), leaf)
        if eqx.is_array(leaf) and np.issubdtype(leaf.dtype, np.floating):
            return leaf.astype(jnp.float64)
        return leaf

    return jt.map(replace_numeric_scalar, tree)


def legacy_static_numeric_filter(file_obj: Any, leaf: Any) -> Any:
    """Read historical numeric placeholders while preserving template shape."""

    if eqx.is_array(leaf):
        out = jnp.load(file_obj)
        if leaf.ndim == 1 and out.shape == ():
            out = jnp.full(leaf.shape, out, dtype=leaf.dtype)
        if out.dtype != leaf.dtype:
            out = out.astype(leaf.dtype)
        return out
    return eqx.default_deserialise_filter_spec(file_obj, leaf)


def scalar_from_legacy_array(value: Any, scalar_type: type) -> Any:
    """Return the first scalar from a historical replicated scalar array."""

    raw = np.asarray(value).reshape(-1)[0]
    if scalar_type is bool:
        return bool(raw)
    return scalar_type(raw)


def restore_legacy_static_numeric_metadata(model: Any) -> Any:
    """Restore known historical scalar metadata fields after loading."""

    return eqx.tree_at(
        lambda tree: (
            tree.nodes["efferent"].delay,
            tree.nodes["efferent"].noise_func.terms[0].noise_func.std,
            tree.nodes["efferent"].noise_func.terms[0].noise_func.mean,
            tree.nodes["efferent"].noise_func.terms[1].std,
            tree.nodes["efferent"].noise_func.terms[1].mean,
            tree.nodes["efferent"].add_noise,
            tree.nodes["efferent"].init_value,
            tree.nodes["mechanics"].dt,
            tree.nodes["net"].net.input_size,
            tree.nodes["net"].net.hidden_size,
            tree.nodes["net"].net.out_size,
            tree.nodes["net"].net.population_structure.n_input_only,
            tree.nodes["net"].net.population_structure.n_readout_only,
            tree.nodes["net"].net.population_structure.n_recurrent_only,
            tree.nodes["net"].net.population_structure.n_input_readout,
            tree.nodes["sensory"].delay,
            tree.nodes["sensory"].noise_func.std,
            tree.nodes["sensory"].noise_func.mean,
            tree.nodes["sensory"].add_noise,
            tree.nodes["sensory"].init_value,
        ),
        model,
        (
            scalar_from_legacy_array(model.nodes["efferent"].delay, int),
            scalar_from_legacy_array(
                model.nodes["efferent"].noise_func.terms[0].noise_func.std, float
            ),
            scalar_from_legacy_array(
                model.nodes["efferent"].noise_func.terms[0].noise_func.mean, float
            ),
            scalar_from_legacy_array(model.nodes["efferent"].noise_func.terms[1].std, float),
            scalar_from_legacy_array(model.nodes["efferent"].noise_func.terms[1].mean, float),
            scalar_from_legacy_array(model.nodes["efferent"].add_noise, bool),
            scalar_from_legacy_array(model.nodes["efferent"].init_value, float),
            scalar_from_legacy_array(model.nodes["mechanics"].dt, float),
            scalar_from_legacy_array(model.nodes["net"].net.input_size, int),
            scalar_from_legacy_array(model.nodes["net"].net.hidden_size, int),
            scalar_from_legacy_array(model.nodes["net"].net.out_size, int),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_input_only, int
            ),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_readout_only, int
            ),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_recurrent_only, int
            ),
            scalar_from_legacy_array(
                model.nodes["net"].net.population_structure.n_input_readout, int
            ),
            scalar_from_legacy_array(model.nodes["sensory"].delay, int),
            scalar_from_legacy_array(model.nodes["sensory"].noise_func.std, float),
            scalar_from_legacy_array(model.nodes["sensory"].noise_func.mean, float),
            scalar_from_legacy_array(model.nodes["sensory"].add_noise, bool),
            scalar_from_legacy_array(model.nodes["sensory"].init_value, float),
        ),
    )


def checkpoint_model_template(
    hps: Any,
    seed: int,
    *,
    model_factory: ModelFactory = _default_model_factory,
) -> Any:
    """Build the model template shared by the archival checkpoint readers."""

    return model_factory(hps, jr.PRNGKey(seed))


def load_trained_model_compatible(
    path: Path,
    hps: Any,
    seed: int,
    *,
    run_id: str = "unknown",
    model_factory: ModelFactory = _default_model_factory,
) -> Any:
    """Load a final model, falling back to the historical checkpoint shape."""

    def base_template(key: Any) -> Any:
        return model_factory(hps, key)

    try:
        model, _hyperparameters = load_with_hyperparameters(
            path, setup_func=lambda key, **_kwargs: base_template(key)
        )
        return model
    except Exception:
        n_replicates = int(hps.model.n_replicates)

        def legacy_template(key: Any) -> Any:
            return legacy_static_numeric_template(
                force_legacy_masked_readout(base_template(key)), n_replicates
            )

        try:
            model, _hyperparameters = load_with_hyperparameters(
                path,
                setup_func=lambda key, **_kwargs: legacy_template(key),
                filter_spec=legacy_static_numeric_filter,
            )
        except Exception as legacy_error:
            raise RuntimeError(
                f"Could not load trained model for {run_id!r} with either normal "
                "or legacy static-numeric compatibility templates."
            ) from legacy_error
        try:
            return restore_legacy_static_numeric_metadata(model)
        except Exception as restore_error:
            raise RuntimeError(
                f"Loaded legacy model for {run_id!r} but could not restore scalar metadata."
            ) from restore_error


def load_checkpoint_model_compatible(
    path: Path,
    hps: Any,
    seed: int,
    *,
    model_factory: ModelFactory = _default_model_factory,
) -> Any:
    """Load a checkpoint with the final-model historical reconstruction."""

    template = checkpoint_model_template(hps, seed, model_factory=model_factory)
    try:
        return eqx.tree_deserialise_leaves(path, template)
    except Exception:
        legacy_template = legacy_static_numeric_template(
            force_legacy_masked_readout(template), int(hps.model.n_replicates)
        )
        model = eqx.tree_deserialise_leaves(
            path, legacy_template, filter_spec=legacy_static_numeric_filter
        )
        return restore_legacy_static_numeric_metadata(model)


def load_nonh0_checkpoint_model_compatible(
    path: Path,
    hps: Any,
    seed: int,
    *,
    model_factory: ModelFactory = _default_model_factory,
) -> Any:
    """Load a non-H0 SimpleStagedNetwork checkpoint with float64 leaves."""

    def cast_floating_leaf(leaf: Any) -> Any:
        if eqx.is_array(leaf) and np.issubdtype(leaf.dtype, np.floating):
            return leaf.astype(jnp.float64)
        return leaf

    template = checkpoint_model_template(hps, seed, model_factory=model_factory)
    model = eqx.tree_deserialise_leaves(path, jt.map(cast_floating_leaf, template))
    net = model.nodes["net"]
    if getattr(net, "dtype", None) is not jnp.float64:
        model = eqx.tree_at(lambda tree: tree.nodes["net"].dtype, model, jnp.float64)
        net = model.nodes["net"]
    if hasattr(net, "_initial_state"):
        model = eqx.tree_at(
            lambda tree: tree.nodes["net"]._initial_state,
            model,
            jt.map(cast_floating_leaf, net._initial_state),
        )
    return model
