"""Loaders for minimax-trained-checkpoint directories.

The minimax adversarial trainer (``scripts/train_minimax.py``) writes a
specific on-disk layout into each ``_artifacts/<exp>/runs/<run>/`` directory:

- ``config.json`` — run hyperparameters (the argparse namespace)
- ``warmup_model.eqx`` — model after the warm-start phase (pre-adversarial)
- ``adversarial_model.eqx`` — model after the adversarial phase
- ``trained_adversary.eqx`` — saved :class:`rlrmp.adversary.GaussianBumpAdversary`

These helpers load any of those artifacts. They are training-method-specific
(a minimax run dir has different on-disk semantics than a standard run dir),
hence the dedicated module.

Bug: 8404108 — extracted from ``scripts/eval_minimax.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
from equinox.nn import StateIndex
from feedbax.intervene import FixedFieldParams
from jax_cookbook import load_with_hyperparameters

from rlrmp.adversary import GaussianBumpAdversary
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.train.task_model import setup_task_model_pair

__all__ = ["load_adversary", "load_config", "load_model"]


def load_config(results_dir: Path) -> dict:
    """Load ``config.json`` from a minimax results directory.

    Args:
        results_dir: Directory containing the minimax-run output (typically
            under ``_artifacts/<exp>/runs/<run>/``).

    Returns:
        Parsed ``config.json`` dict.

    Raises:
        FileNotFoundError: if ``config.json`` is missing.
    """
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {results_dir}")
    with open(config_path) as f:
        return json.load(f)


def _squeeze_replicate_axis(model):
    """Remove a leading singleton replicate axis (shape ``[1, ...] → [...]``).

    ``train_minimax.py`` with ``n_replicates=1`` saves models with a leading
    ``[1, ...]`` axis. For evaluation we want a single model without that axis.
    """
    return jt.map(
        lambda x: x[0] if (hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == 1) else x,
        model,
        is_leaf=eqx.is_array,
    )


def make_legacy_minimax_model_template(hps, *, key):
    """Build the legacy float32/int32 minimax template used by archived runs."""

    template = setup_task_model_pair(hps, key=key).model
    template = _with_legacy_intervenor_state_index(template)
    return jtu.tree_map_with_path(_legacy_serialized_array_leaf, template)


def normalize_loaded_minimax_runtime(model, hps, *, key, current_model=None):
    """Restore current runtime state dtypes after legacy checkpoint materialization."""

    current = setup_task_model_pair(hps, key=key).model if current_model is None else current_model
    if "mechanics" in model.nodes and "mechanics" in current.nodes:
        model = eqx.tree_at(
            lambda m: m.nodes["mechanics"].state_index.init,
            model,
            current.nodes["mechanics"].state_index.init,
        )
    if "feedback" in model.nodes and "feedback" in current.nodes:
        model = eqx.tree_at(
            lambda m: m.nodes["feedback"].channels.state_index.init,
            model,
            current.nodes["feedback"].channels.state_index.init,
        )
    if "force_filter" in model.nodes and "force_filter" in current.nodes:
        model = eqx.tree_at(
            lambda m: m.nodes["force_filter"].state_index.init,
            model,
            current.nodes["force_filter"].state_index.init,
        )

    node = model.nodes.get(PLANT_INTERVENOR_LABEL)
    if node is not None and hasattr(node, "params_index"):
        params = node.params_index.init
        model = eqx.tree_at(
            lambda m: (
                m.nodes[PLANT_INTERVENOR_LABEL].params_index.init.scale,
                m.nodes[PLANT_INTERVENOR_LABEL].params_index.init.active,
                m.nodes[PLANT_INTERVENOR_LABEL].params_index.init.amplitude,
            ),
            model,
            (
                jnp.asarray(params.scale, dtype=jnp.float32),
                jnp.asarray(params.active, dtype=jnp.bool_),
                jnp.asarray(params.amplitude, dtype=jnp.float32),
            ),
        )
    return model


def _with_legacy_intervenor_state_index(model):
    node = model.nodes.get(PLANT_INTERVENOR_LABEL)
    if node is None or not hasattr(node, "params_index"):
        return model

    params = FixedFieldParams()
    object.__setattr__(params, "scale", 1.0)
    object.__setattr__(params, "active", False)
    object.__setattr__(params, "amplitude", 1.0)
    object.__setattr__(params, "field", jnp.zeros((2,), dtype=jnp.float32))

    state_index = StateIndex(params)
    object.__setattr__(state_index, "marker", node.params_index.marker)
    return eqx.tree_at(lambda m: m.nodes[PLANT_INTERVENOR_LABEL].params_index, model, state_index)


def _legacy_serialized_array_leaf(path, leaf):
    _ = path
    if not eqx.is_array(leaf):
        return leaf
    if jnp.issubdtype(leaf.dtype, jnp.floating):
        return jnp.asarray(leaf, dtype=jnp.float32)
    if jnp.issubdtype(leaf.dtype, jnp.integer):
        return jnp.asarray(leaf, dtype=jnp.int32)
    return leaf


def load_model(results_dir: Path, filename: str, hps, config: dict):
    """Load a model by ``filename`` from ``results_dir``.

    Tries two templates: first with the unsqueezed ``[1, ...]`` replicate axis
    (``warmup_model.eqx`` is saved directly from ``train_pair``), then with a
    squeezed template (``adversarial_model.eqx`` is saved after squeezing).
    Always returns the model with any replicate axis removed.

    Args:
        results_dir: Run directory containing the model file.
        filename: Model filename (e.g. ``"warmup_model.eqx"``,
            ``"adversarial_model.eqx"``).
        hps: Hyperparameter ``TreeNamespace`` (typically from
            ``rlrmp.train.minimax.build_hps``).
        config: Parsed ``config.json`` (unused at present but kept for
            future template-construction overrides).

    Returns:
        The loaded model (squeezed), or ``None`` if the file is not present.

    Raises:
        RuntimeError: if both squeezed and unsqueezed templates fail.
    """
    _ = config  # kept for forward compatibility / API symmetry
    model_path = results_dir / filename
    if not model_path.exists():
        return None

    def _make_template(key):
        return setup_task_model_pair(hps, key=key).model

    def _make_squeezed_template(key):
        template = setup_task_model_pair(hps, key=key).model
        return _squeeze_replicate_axis(template)

    def _make_legacy_template(key):
        return make_legacy_minimax_model_template(hps, key=key)

    errors: list[tuple[str, Exception]] = []

    # Try unsqueezed template first (warmup model), then squeezed (adversarial model).
    for label, make_template, already_squeezed in [
        ("unsqueezed", _make_template, False),
        ("squeezed", _make_squeezed_template, True),
        ("legacy_minimax", _make_legacy_template, False),
    ]:
        try:
            model, _ = load_with_hyperparameters(
                model_path,
                setup_func=lambda key, **kwargs: make_template(key),
            )
            current_model = setup_task_model_pair(hps, key=jr.PRNGKey(0)).model
            if already_squeezed:
                current_model = _squeeze_replicate_axis(current_model)
            model = normalize_loaded_minimax_runtime(
                model,
                hps,
                key=jr.PRNGKey(0),
                current_model=current_model,
            )
            # If loaded with unsqueezed template, squeeze it now for eval.
            if not already_squeezed:
                model = _squeeze_replicate_axis(model)
            return model
        except (RuntimeError, ValueError) as exc:
            errors.append((label, exc))
            continue

    details = "; ".join(f"{label}: {type(exc).__name__}: {exc}" for label, exc in errors)
    raise RuntimeError(
        f"Could not load model from {model_path} with current or legacy templates. {details}"
    )


def load_adversary(results_dir: Path, hps) -> GaussianBumpAdversary | None:
    """Load the trained adversary from ``results_dir/trained_adversary.eqx``.

    Args:
        results_dir: Run directory containing ``trained_adversary.eqx``.
        hps: Hyperparameter ``TreeNamespace``. Used to size the template
            (``hps.task.n_steps - 1`` timesteps, ``hps.dt``).

    Returns:
        Loaded :class:`GaussianBumpAdversary`, or ``None`` if the file is
        absent or deserialisation fails (a warning is printed in the latter
        case so the caller can fall back to pre-saved force profiles).
    """
    adv_path = results_dir / "trained_adversary.eqx"
    if not adv_path.exists():
        return None

    n_timesteps = hps.task.n_steps - 1
    adversary_template = GaussianBumpAdversary(
        n_bumps=3,  # default; overridden by loaded weights
        n_timesteps=n_timesteps,
        n_force_dims=2,
        force_max=1.0,
        dt=hps.dt,
        key=jr.PRNGKey(0),
    )
    try:
        adversary = eqx.tree_deserialise_leaves(adv_path, adversary_template)
    except Exception as e:
        print(
            f"WARNING: could not deserialise adversary ({e}); will use pre-saved force profiles only."
        )
        return None
    return adversary
