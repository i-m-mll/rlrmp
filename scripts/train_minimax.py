"""Minimax adversarial training script for the RLRMP project.

Phase 1 (warm-start): Train the controller normally with random gust perturbations.
Phase 2 (adversarial): Alternate between adversary gradient ascent and controller gradient descent.
Supports fused mode (--fused, default) which compiles the K adversary steps + 1 controller
step into a single JIT call via lax.fori_loop, and decomposed mode (--no-fused) which uses
K×2 + 1 separate JIT calls per batch.

The adversary (GaussianBumpAdversary) generates SISU-conditional force profiles that
replace random gusts during adversarial training.

Population-based mode (--n-adversaries K) creates K independent adversaries that
rotate each batch (adversary index = batch_idx % K), providing diverse perturbation
pressure. When K=1, behavior is identical to the original single-adversary mode.

Usage:
    uv run python scripts/train_minimax.py --n-warmup-batches 2000 --n-adversary-batches 8000
    uv run python scripts/train_minimax.py --n-warmup-batches 20 --n-adversary-batches 30 \
        --output-dir /tmp/minimax_smoke
    uv run python scripts/train_minimax.py --n-adversaries 5 --n-adversary-batches 10000 \
        --output-dir results/pop_adversary
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax.tree_util as jtu
import numpy as np
import optax
from feedbax import prepare_trial
from jax_cookbook import load_with_hyperparameters
from jax_cookbook import save as fbx_save
from feedbax.runtime.iteration import run_component
from feedbax.runtime.batch import BatchInfo
from feedbax.objectives.streaming import make_streaming_loss_fn
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training.checkpoint_custody import CheckpointCompatibilityError
from feedbax.training.train import (
    TaskTrainer,
    make_delayed_cosine_schedule,
    train_pair,
)

from rlrmp.train.adversarial_training import (
    _inject_adversary_delta_A,
    _inject_adversary_forces,
)
from rlrmp.train.adversary import GaussianBumpAdversary, LinearDynamicsAdversary
from rlrmp.intervention_compat import LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
from rlrmp.model.feedbax_graph import (
    build_runtime_rlrmp_feedbax_graph_bundle,
    write_graph_spec_bundle,
)
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path
from rlrmp.model.trainable import staged_network_trainable_parts
from rlrmp.train.progress import (
    batch_log_every,
    format_batch_line,
    make_batch_log_callbacks,
    should_log_batch,
)

# build_hps was extracted to rlrmp.train.minimax in 8404108 (capability-named
# library module; previously defined inline here and pulled by analysis scripts
# via sys.path injection). Re-imported for internal use; analysis / eval scripts
# should import from `rlrmp.train` directly.
from rlrmp.train.minimax import (  # noqa: F401  (re-exported intentionally)
    build_hps,
    build_minimax_training_run_spec,
    legacy_cli_args_to_minimax_config,
    minimax_config_namespace,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.runtime.checkpoint_custody import (
    MINIMAX_ADVERSARIAL_BARRIER,
    MINIMAX_WARMUP_BARRIER,
    deserialize_pytree_slot,
    has_custody_checkpoint,
    load_minimax_checkpoint_transaction,
    serialize_pytree_slot,
    spec_digests,
    write_minimax_checkpoint_transaction,
)

logger = logging.getLogger(__name__)


def _adversary_update(adversary_optimizer, adversary, dL_dforces, adv_opt_st):
    """Update adversary parameters via VJP through adversary().

    Performs gradient ascent: adversary maximises loss, so we negate dL_dforces
    before computing the VJP (equivalent to ascending the loss gradient).

    Args:
        adversary_optimizer: Optax optimizer for the adversary (static).
        adversary: Current GaussianBumpAdversary.
        dL_dforces: Gradient of loss w.r.t. forces, shape (batch_size, T, d).
        adv_opt_st: Current adversary optimizer state.

    Returns:
        Tuple of (updated_adversary, updated_opt_state).
    """

    # forces = adversary() broadcast to (batch_size, T, d)
    # We need dL/d(adversary_params) = dL/dforces * dforces/d(adversary_params)
    # Use jax.linear_util / vjp directly through the broadcast.
    def _forces_fn(a):
        force_profile = a()  # (T, d)
        return jnp.broadcast_to(force_profile, dL_dforces.shape)

    # VJP: dL/d(params) via chain rule through the broadcast
    _, vjp_fn = jax.vjp(lambda a: eqx.filter(_forces_fn(a), eqx.is_array), adversary)
    # Negate for gradient ascent
    neg_dL_dforces = jt.map(lambda g: -g, dL_dforces)
    (param_grads,) = vjp_fn(neg_dL_dforces)

    updates, new_opt_st = adversary_optimizer.update(
        eqx.filter(param_grads, eqx.is_array),
        adv_opt_st,
        eqx.filter(adversary, eqx.is_array),
    )
    new_adversary = eqx.apply_updates(adversary, updates)
    return new_adversary, new_opt_st


# ---------------------------------------------------------------------------
# Spec-dir / artifact-dir helpers
# ---------------------------------------------------------------------------


def derive_spec_dir(output_dir: Path) -> Path:
    """Derive the tracked run sidecar/spec directory from the artifact directory.

    Applies the mirror invariant ``run_artifact_dir(exp, run)`` ↔
    ``run_spec_sidecar_dir(exp, run)``: paths under ``<repo>/_artifacts/...``
    are re-rooted under ``<repo>/results/...``. Paths outside the
    ``_artifacts/`` tree fall back to a sibling ``<output_dir>_spec``.

    Args:
        output_dir: Absolute or relative path to the bulk-artifact directory
            (typically under ``_artifacts/<exp>/runs/<run>/``).

    Returns:
        Absolute path to the corresponding sidecar/spec directory.
    """
    out = Path(output_dir).resolve()
    artifact_root = (REPO_ROOT / "_artifacts").resolve()
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        return out.parent / (out.name + "_spec")


def derive_spec_path(output_dir: Path) -> Path:
    """Derive the canonical FLAT tracked run-recipe file from the artifact dir.

    The run recipe lives at the flat ``results/<exp>/runs/<run>.json`` path
    (CLAUDE.md §Run-folder convention), NOT the legacy nested
    ``results/<exp>/runs/<run>/run.json`` form. For artifact paths shaped like
    ``_artifacts/<exp>/runs/<run>/`` this returns
    ``rlrmp.paths.run_spec_path(exp, run)``; out-of-tree paths fall back to a
    sibling ``<output_dir>.json`` file alongside the artifacts.

    Args:
        output_dir: Path to the bulk-artifact directory (typically under
            ``_artifacts/<exp>/runs/<run>/``).

    Returns:
        Absolute path to the flat ``<run>.json`` recipe file.
    """
    sidecar_dir = derive_spec_dir(output_dir)
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = sidecar_dir.resolve().relative_to(spec_root)
    except ValueError:
        # Out-of-tree fallback: flat <output_dir>.json sibling.
        return sidecar_dir.parent / (sidecar_dir.name + ".json")
    parts = rel.parts
    # Expect <exp>/runs/<run>; map to the flat run_spec_path(exp, run).
    # for_write=True forces the canonical flat path with no legacy fallback so
    # re-training never overwrites a stale nested legacy recipe (W8/e926665).
    if len(parts) == 3 and parts[1] == "runs":
        return run_spec_path(parts[0], parts[2], for_write=True)
    # Unexpected shape: place a flat <name>.json next to the sidecar dir.
    return sidecar_dir.parent / (sidecar_dir.name + ".json")


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def _get_git_metadata() -> dict:
    """Capture git info for reproducibility."""
    meta = {}
    try:
        import rlrmp

        meta["rlrmp_version"] = getattr(rlrmp, "__version__", "unknown")
    except ImportError:
        pass
    for cmd, key in [
        (["git", "rev-parse", "HEAD"], "rlrmp_commit"),
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "rlrmp_branch"),
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                meta[key] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    try:
        import jax

        meta["jax_version"] = jax.__version__
    except ImportError:
        pass
    try:
        import feedbax

        meta["feedbax_version"] = getattr(feedbax, "__version__", "unknown")
    except ImportError:
        pass
    return meta


def _collect_gpu_info() -> dict:
    """Capture GPU/device info for run-config reproducibility.

    Records JAX-visible device kinds, count, and (best-effort) per-device
    total memory in GiB via ``nvidia-smi``. Never raises — failures are
    caught and surfaced as ``nvidia_smi_error`` / ``device_memory_gb_total =
    None`` so the surrounding training run is not blocked. Bug: c723082.

    Returns:
        Dict with keys ``device_kinds``, ``device_count``, optionally
        ``device_memory_gb_total`` and ``nvidia_smi_error``. Safe to embed
        directly in the run-config JSON written to disk.
    """
    info: dict = {}
    try:
        devices = jax.devices()
        info["device_kinds"] = [d.device_kind for d in devices]
        info["device_count"] = len(devices)
    except Exception as e:
        info["device_kinds"] = None
        info["device_count"] = 0
        info["jax_devices_error"] = str(e)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        mem_csv = result.stdout.strip().split("\n")
        info["device_memory_gb_total"] = [float(m) / 1024.0 for m in mem_csv if m]
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
        info["device_memory_gb_total"] = None
        info["nvidia_smi_error"] = str(e)
    except Exception as e:  # pragma: no cover — defensive
        info["device_memory_gb_total"] = None
        info["nvidia_smi_error"] = f"unexpected: {e}"
    return info


def _configure_jax_runtime(args) -> None:
    """Configure JAX runtime options that must be set before first compile."""
    cache_dir = args.jax_cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_path))
        logger.info("Using JAX compilation cache dir: %s", cache_path)

    if args.jax_explain_cache_misses:
        jax.config.update("jax_explain_cache_misses", True)
        logger.info("Enabled jax_explain_cache_misses for cache diagnostics")


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR_NAME = "checkpoints_adversarial"
_CHECKPOINT_SUBDIR = "checkpoint_latest"


def _save_adversarial_checkpoint(
    checkpoint_dir: Path,
    flat_model: list,
    treedef_model,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state,
    batch_idx: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
    *,
    training_spec: TrainingRunSpec | None = None,
    rng_key=None,
) -> None:
    """Save adversarial training state to custody and a materialized directory.

    Writes atomically: assembles state in a temp dir then renames it over
    the previous checkpoint so a preempted write never leaves corrupt state.

    The model is serialized with ``eqx.tree_serialise_leaves`` (template needed
    at load time is the current ``adv_model`` reconstructed via ``treedef_model``).
    Adversaries and optimizer states are similarly serialized.

    Args:
        checkpoint_dir: Parent directory; ``checkpoint_latest/`` is created inside.
        flat_model: Flat list of model array leaves (from ``jtu.tree_flatten``).
        treedef_model: PyTree treedef for reconstructing the model.
        adversaries: List of ``GaussianBumpAdversary`` instances.
        adv_opt_states: List of adversary optimizer states (one per adversary).
        ctrl_opt_state: Controller optimizer state.
        batch_idx: Index of the batch that was just completed.
        adv_losses: Adversary loss history (up to and including ``batch_idx``).
        ctrl_losses: Controller loss history.
        adv_indices: Active adversary index history.
    """
    model = jtu.tree_unflatten(treedef_model, flat_model)
    if training_spec is not None:
        write_minimax_checkpoint_transaction(
            checkpoint_dir,
            training_spec=training_spec,
            barrier_name=MINIMAX_ADVERSARIAL_BARRIER,
            batch_idx=batch_idx,
            active_member_index=adv_indices[-1] if adv_indices else -1,
            slots=_minimax_checkpoint_slots(
                model=model,
                adversaries=adversaries,
                adv_opt_states=adv_opt_states,
                ctrl_opt_state=ctrl_opt_state,
                rng_key=rng_key,
                batch_idx=batch_idx,
                active_member_index=adv_indices[-1] if adv_indices else -1,
                adv_losses=adv_losses,
                ctrl_losses=ctrl_losses,
                adv_indices=adv_indices,
                training_spec=training_spec,
            ),
            population_member_ids=_adversary_population_member_ids(adversaries),
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / _CHECKPOINT_SUBDIR

    # Write into a sibling temp dir for atomic replacement.
    tmp_dir = checkpoint_dir / (f"_{_CHECKPOINT_SUBDIR}_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    try:
        # 1. Model: reconstruct → serialize leaves
        eqx.tree_serialise_leaves(tmp_dir / "model.eqx", model)

        # 2. Adversaries: serialize each
        for i, adv in enumerate(adversaries):
            eqx.tree_serialise_leaves(tmp_dir / f"adversary_{i}.eqx", adv)

        # 3. Optimizer states: serialize
        eqx.tree_serialise_leaves(tmp_dir / "ctrl_opt_state.eqx", ctrl_opt_state)
        for i, opt_st in enumerate(adv_opt_states):
            eqx.tree_serialise_leaves(tmp_dir / f"adv_opt_state_{i}.eqx", opt_st)

        # 4. Scalar progress + loss histories
        meta = {
            "batch_idx": batch_idx,
            "n_adversaries": len(adversaries),
            "adv_losses": adv_losses,
            "ctrl_losses": ctrl_losses,
            "adv_indices": adv_indices,
        }
        with open(tmp_dir / "meta.json", "w") as fh:
            json.dump(meta, fh)

        # Atomic rename
        if target.exists():
            shutil.rmtree(target)
        tmp_dir.rename(target)

    except Exception:
        # Leave the tmp dir for debugging; do not corrupt the previous checkpoint.
        logger.exception("Checkpoint save failed — previous checkpoint (if any) is intact")
        raise


def _load_adversarial_checkpoint(
    checkpoint_dir: Path,
    model_template,
    adversaries_template: list,
    adv_opt_states_template: list,
    ctrl_opt_state_template,
    treedef_model,
    *,
    training_spec: TrainingRunSpec | None = None,
):
    """Load adversarial training state from ``checkpoint_latest/``.

    Args:
        checkpoint_dir: Parent directory containing ``checkpoint_latest/``.
        model_template: Model object with the correct PyTree structure (template).
        adversaries_template: List of ``GaussianBumpAdversary`` instances with the
            correct structure (used as deserialization templates).
        adv_opt_states_template: List of adversary optimizer states (templates).
        ctrl_opt_state_template: Controller optimizer state (template).
        treedef_model: PyTree treedef for the model (used to re-flatten the result).

    Returns:
        Tuple of ``(flat_model, adversaries, adv_opt_states, ctrl_opt_state,
        resume_batch_idx, adv_losses, ctrl_losses, adv_indices)``.
    """
    if training_spec is not None and has_custody_checkpoint(checkpoint_dir):
        loaded = load_minimax_checkpoint_transaction(
            checkpoint_dir,
            training_spec=training_spec,
            expected_slots=_minimax_expected_slots(
                model_template=model_template,
                adversaries_template=adversaries_template,
                adv_opt_states_template=adv_opt_states_template,
                ctrl_opt_state_template=ctrl_opt_state_template,
            ),
            expected_population_member_ids=_adversary_population_member_ids(adversaries_template),
        )
        slots = loaded.slots
        try:
            controller = deserialize_pytree_slot(
                slots["controller"],
                model_template,
                slot="controller",
            )
            adversaries = [
                deserialize_pytree_slot(blob, template, slot="adversary_population")
                for blob, template in zip(
                    slots["adversary_population"],
                    adversaries_template,
                    strict=True,
                )
            ]
            adv_opt_states = deserialize_pytree_slot(
                slots["adversary_optimizer"],
                adv_opt_states_template,
                slot="adversary_optimizer",
            )
            ctrl_opt_state = deserialize_pytree_slot(
                slots["controller_optimizer"],
                ctrl_opt_state_template,
                slot="controller_optimizer",
            )
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "minimax checkpoint PyTree slot could not be deserialized with the resume template"
            ) from exc
        if len(adversaries) != len(adversaries_template):
            raise CheckpointCompatibilityError(
                "population identity mismatch for slot 'adversary_population'"
            )
        flat_model = jtu.tree_flatten(controller)[0]
        return (
            flat_model,
            adversaries,
            adv_opt_states,
            ctrl_opt_state,
            int(slots["active_batch_index"]),
            list(slots.get("adversary_losses", [])),
            list(slots.get("controller_losses", [])),
            list(slots.get("adversary_indices", [])),
            jnp.asarray(slots["rng"], dtype=jnp.uint32),
        )

    target = checkpoint_dir / _CHECKPOINT_SUBDIR
    if not target.exists():
        raise FileNotFoundError(f"No checkpoint found at {target}")

    # 1. Model — deserialize onto the template; then flatten for the training loop
    model = eqx.tree_deserialise_leaves(target / "model.eqx", model_template)
    flat_model = jtu.tree_flatten(model)[0]

    # 2. Adversaries
    adversaries = []
    for i, tmpl in enumerate(adversaries_template):
        adv = eqx.tree_deserialise_leaves(target / f"adversary_{i}.eqx", tmpl)
        adversaries.append(adv)

    # 3. Optimizer states
    ctrl_opt_state = eqx.tree_deserialise_leaves(
        target / "ctrl_opt_state.eqx", ctrl_opt_state_template
    )
    adv_opt_states = []
    for i, tmpl in enumerate(adv_opt_states_template):
        opt_st = eqx.tree_deserialise_leaves(target / f"adv_opt_state_{i}.eqx", tmpl)
        adv_opt_states.append(opt_st)

    # 4. Meta
    with open(target / "meta.json") as fh:
        meta = json.load(fh)

    return (
        flat_model,
        adversaries,
        adv_opt_states,
        ctrl_opt_state,
        meta["batch_idx"],
        meta["adv_losses"],
        meta["ctrl_losses"],
        meta["adv_indices"],
    )


def _write_warmup_boundary_checkpoint(
    checkpoint_dir: Path,
    *,
    training_spec: TrainingRunSpec,
    model,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state,
    rng_key,
    warmup_history,
) -> None:
    write_minimax_checkpoint_transaction(
        checkpoint_dir,
        training_spec=training_spec,
        barrier_name=MINIMAX_WARMUP_BARRIER,
        batch_idx=-1,
        active_member_index=-1,
        slots=_minimax_checkpoint_slots(
            model=model,
            adversaries=adversaries,
            adv_opt_states=adv_opt_states,
            ctrl_opt_state=ctrl_opt_state,
            rng_key=rng_key,
            batch_idx=-1,
            active_member_index=-1,
            adv_losses=[],
            ctrl_losses=[],
            adv_indices=[],
            training_spec=training_spec,
            warmup_history=warmup_history,
        ),
        population_member_ids=_adversary_population_member_ids(adversaries),
    )


def _write_final_minimax_custody_transaction(
    checkpoint_dir: Path,
    *,
    training_spec: TrainingRunSpec,
    model,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state,
    rng_key,
    batch_idx: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
    warmup_history=None,
) -> None:
    """Write the terminal (``status="final"``) minimax custody transaction.

    This is the content-addressed durable authority for the run's final
    outputs — the trained controller, adversary population, loss curves, and
    warmup history — published through the feedbax custody latest-pointer run
    record. The ``output_dir`` ``.eqx`` / ``.npz`` files written alongside are
    compatibility materializations, not the durable record. Issue 7e71950.
    """
    active_member_index = adv_indices[-1] if adv_indices else -1
    write_minimax_checkpoint_transaction(
        checkpoint_dir,
        training_spec=training_spec,
        barrier_name=MINIMAX_ADVERSARIAL_BARRIER,
        batch_idx=batch_idx,
        active_member_index=active_member_index,
        slots=_minimax_checkpoint_slots(
            model=model,
            adversaries=adversaries,
            adv_opt_states=adv_opt_states,
            ctrl_opt_state=ctrl_opt_state,
            rng_key=rng_key,
            batch_idx=batch_idx,
            active_member_index=active_member_index,
            adv_losses=adv_losses,
            ctrl_losses=ctrl_losses,
            adv_indices=adv_indices,
            training_spec=training_spec,
            warmup_history=warmup_history,
        ),
        population_member_ids=_adversary_population_member_ids(adversaries),
        status="final",
    )


def _minimax_checkpoint_slots(
    *,
    model,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state,
    rng_key,
    batch_idx: int,
    active_member_index: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
    training_spec: TrainingRunSpec,
    warmup_history=None,
) -> dict[str, object]:
    slots: dict[str, object] = {
        "controller": serialize_pytree_slot(model),
        "controller_optimizer": serialize_pytree_slot(ctrl_opt_state),
        "adversary_population": [serialize_pytree_slot(adversary) for adversary in adversaries],
        "adversary_optimizer": serialize_pytree_slot(adv_opt_states),
        "rng": jnp.asarray(rng_key, dtype=jnp.uint32),
        "active_batch_index": jnp.asarray(batch_idx, dtype=jnp.int32),
        "active_member_index": jnp.asarray(active_member_index, dtype=jnp.int32),
        "spec_digests": spec_digests(training_spec),
    }
    if adv_losses:
        slots["adversary_losses"] = list(adv_losses)
    if ctrl_losses:
        slots["controller_losses"] = list(ctrl_losses)
    if adv_indices:
        slots["adversary_indices"] = list(adv_indices)
    if warmup_history is not None:
        slots["warmup_history"] = warmup_history
    return slots


def _minimax_expected_slots(
    *,
    model_template,
    adversaries_template: list,
    adv_opt_states_template: list,
    ctrl_opt_state_template,
) -> dict[str, object]:
    return {
        "rng": jnp.asarray([0, 0], dtype=jnp.uint32),
        "active_batch_index": jnp.asarray(0, dtype=jnp.int32),
        "active_member_index": jnp.asarray(0, dtype=jnp.int32),
    }


def _adversary_population_member_ids(adversaries: list) -> dict[str, list[str]]:
    return {"adversary_population": [f"adversary_{index}" for index, _ in enumerate(adversaries)]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _get_trainable(model):
    """Return the trainable leaves of the model.

    Default (SimpleStagedNetwork): (net.hidden, net.readout). Native affine
    linear variants carry their parameters as ``gain`` and optional
    ``feedforward`` directly on the net Component — branch on Module class
    name to keep this single function compatible with both code paths.
    """
    net = model.get_node("net")
    cls_name = type(net).__name__
    if cls_name == "AffineFeedbackController":
        if getattr(net, "feedforward", None) is not None:
            return model.get_node_attrs("net", "gain", "feedforward")
        return model.get_node_attrs("net", "gain")
    if cls_name == "LinearController":
        return model.get_node_attrs("net", "K")
    if cls_name == "LinearTrackerController":
        return model.get_node_attrs("net", "K", "u_ff")
    return staged_network_trainable_parts(net)


def _trainable_where(model):
    """Return the ``eqx.tree_at``-compatible selector lambda for trainable leaves.

    Mirrors ``_get_trainable`` but returns a ``where`` function (mapping a model
    to its trainable subtrees) instead of the subtrees themselves. Required by
    ``eqx.tree_at`` to splice updated parameters back into the model PyTree.
    Bug: 410d7ac — the linear-controller MVP variants do not have
    ``net.hidden`` / ``net.readout`` attributes, so the default selector
    triggers an AttributeError during the adversarial controller step. This
    helper centralises the architecture branch.
    """
    net = model.get_node("net")
    cls_name = type(net).__name__
    if cls_name == "AffineFeedbackController":
        if getattr(net, "feedforward", None) is not None:
            return lambda m: m.get_node_attrs("net", "gain", "feedforward")
        return lambda m: m.get_node_attrs("net", "gain")
    if cls_name == "LinearController":
        return lambda m: m.get_node_attrs("net", "K")
    if cls_name == "LinearTrackerController":
        return lambda m: m.get_node_attrs("net", "K", "u_ff")
    return lambda m: staged_network_trainable_parts(m.get_node("net"))


def _eval_trials_streaming(task, model, trial_specs, keys, loss_func):
    """Evaluate trials with streaming loss — no trajectory stored.

    Mirrors ``task.eval_trials`` but passes a ``streaming_loss_fn`` to
    ``run_component`` so that loss is accumulated inside the scan body.
    Returns the mean scalar loss across the batch (no state history).

    Args:
        task: The task instance (provides intervention_state_indices, etc.).
        model: The model to evaluate.
        trial_specs: Batched trial specifications (leading batch dim).
        keys: Per-trial PRNG keys, shape ``(batch_size,)``.
        loss_func: The loss function (``AbstractLoss``); used to build the
            per-step streaming closure via ``make_streaming_loss_fn``.

    Returns:
        Scalar loss averaged over the batch.
    """

    # Bug: 3ef9c25 — streaming loss integration for memory-efficient training
    def eval_single(trial_spec, key):
        key_run = jr.split(key, 2)[1]
        prepared = prepare_trial(model, trial_spec)
        prepared_inputs = _with_declarative_component_parameter_inputs(
            model,
            prepared.inputs,
        )

        # Build per-step streaming loss closure (single trial, no batch dim)
        streaming_fn = make_streaming_loss_fn(loss_func, trial_spec, model, prepared.n_steps)

        _outputs, _final_state, total_loss = run_component(
            model,
            prepared_inputs,
            prepared.init_state,
            key=key_run,
            n_steps=prepared.n_steps,
            streaming_loss_fn=streaming_fn,
        )
        return total_loss

    per_trial_losses = eqx.filter_vmap(eval_single)(trial_specs, keys)
    return per_trial_losses.mean()


def _with_declarative_component_parameter_inputs(model, inputs):
    """Mirror legacy prepared intervention input under the declared task binding."""
    target = LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
    legacy_key = f"intervene:{target['task_parameter_label']}"
    declared_key = (
        f"task:{target['source_data_id']}->{target['target_node_id']}.{target['target_port']}"
    )
    if (
        isinstance(inputs, dict)
        and legacy_key in inputs
        and declared_key in getattr(model, "input_ports", ())
        and declared_key not in inputs
    ):
        return {**inputs, declared_key: inputs[legacy_key]}
    return inputs


def _make_where_train(sisu_gating: str = "additive"):
    """Return the where_train dict for the controller optimizer."""

    def where_train_fn(model):
        net = model.get_node("net")
        cls_name = type(net).__name__
        # Native affine linear variants carry their parameters directly on the
        # net Component as gain (+ feedforward for the tracker).
        if cls_name == "AffineFeedbackController":
            if getattr(net, "feedforward", None) is not None:
                return model.get_node_attrs("net", "gain", "feedforward")
            return model.get_node_attrs("net", "gain")
        if cls_name == "LinearController":
            return model.get_node_attrs("net", "K")
        if cls_name == "LinearTrackerController":
            return model.get_node_attrs("net", "K", "u_ff")
        return staged_network_trainable_parts(net)

    return {0: where_train_fn}


def run_training(training_run_spec) -> None:
    """Run minimax adversarial training from a validated TrainingRunSpec."""
    if isinstance(training_run_spec, dict) and "feedbax_training_run_spec" in training_run_spec:
        validate_minimax_run_spec(training_run_spec, spec_dir=Path("."))
        feedbax_spec = TrainingRunSpec.model_validate(
            training_run_spec["feedbax_training_run_spec"]
        )
        config_dict = dict(training_run_spec)
    else:
        feedbax_spec = (
            training_run_spec
            if isinstance(training_run_spec, TrainingRunSpec)
            else TrainingRunSpec.model_validate(training_run_spec)
        )
        config_dict = None
    args = minimax_config_namespace(minimax_training_run_spec_to_config(feedbax_spec))
    _configure_jax_runtime(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The canonical run recipe is written to the FLAT
    # results/<exp>/runs/<run>.json path (CLAUDE.md §Run-folder convention).
    # The tracked sidecar directory results/<exp>/runs/<run>/ holds the
    # GraphSpec bundle. When --spec-dir is unset, both are derived from
    # --output-dir via the mirror invariant
    # (paths.run_artifact_dir(exp, run) ↔ paths.run_spec_sidecar_dir(exp, run)).
    # Bug: 0077b42 (sidecar dir), W8/e926665 (flat recipe path).
    spec_dir = Path(args.spec_dir) if args.spec_dir is not None else derive_spec_dir(output_dir)
    mkdir_p(spec_dir)
    spec_path = derive_spec_path(output_dir)
    mkdir_p(spec_path.parent)

    hps = build_hps(args)
    key = jr.PRNGKey(args.seed)
    key_init, key_warmup, key_adv = jr.split(key, 3)

    # -----------------------------------------------------------------------
    # Task / model setup
    # -----------------------------------------------------------------------
    logger.info(
        "Setting up task-model pair (hidden_type=%s, sisu_gating=%s)",
        args.hidden_type,
        args.sisu_gating,
    )
    pair = setup_task_model_pair(hps, key=key_init)
    task = pair.task
    loss_func = task.loss_func

    graph_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair.model)
    graph_path = write_graph_spec_bundle(graph_bundle, spec_dir)
    config_dict = build_minimax_training_run_spec(
        minimax_training_run_spec_to_config(feedbax_spec),
        graph_spec=graph_bundle.graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
        git=_get_git_metadata(),
        gpu_info=_collect_gpu_info(),
        feedbax_graph=graph_bundle.to_run_metadata(
            graph_spec_path=graph_path.name,
        ),
    )
    with open(spec_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info("Saved run spec to %s", spec_path)
    logger.info("Saved Feedbax GraphSpec to %s", graph_path)

    # Build a loss computation closure that abstracts over standard vs
    # streaming evaluation.  Streaming mode accumulates loss inside the scan
    # body, avoiding storage of the full state trajectory.  Bug: 3ef9c25
    use_streaming_loss = args.streaming_loss
    if use_streaming_loss:
        logger.info("Using STREAMING loss (no trajectory storage)")

        def _compute_loss(model, trial_specs, keys):
            return _eval_trials_streaming(task, model, trial_specs, keys, loss_func)
    else:

        def _compute_loss(model, trial_specs, keys):
            states = task.eval_trials(model, trial_specs, keys)
            return loss_func(states, trial_specs, model).total.mean()

    where_train = _make_where_train(sisu_gating=args.sisu_gating)

    # -----------------------------------------------------------------------
    # Phase 1 — warm-start (or load pre-trained model)
    # -----------------------------------------------------------------------
    warmup_model_path = output_dir / "warmup_model.eqx"

    warmup_model = None
    warmup_history = None

    # When resuming, load an already-saved warmup_model.eqx using
    # load_with_hyperparameters and this script's build_hps, since
    # warmup_model.eqx was saved with fbx_save (HDF5 format), not eqx's
    # native format. Using eqx.tree_deserialise_leaves would fail with
    # a TreePathError.
    if args.resume and warmup_model_path.exists():
        logger.info(
            "--resume: loading warmup_model.eqx from %s — skipping phase 1.",
            warmup_model_path,
        )

        def _resume_setup_func(key=jr.PRNGKey(0), **stored_hps):
            """Reconstruct model from stored config for resume."""
            # Filter out non-hps keys that are stored in config.json but not
            # expected by build_hps
            for k in ("git", "output_dir", "checkpoint_every", "resume"):
                stored_hps.pop(k, None)
            resume_args = minimax_config_namespace(stored_hps)
            resume_hps = build_hps(resume_args)
            return setup_task_model_pair(resume_hps, key=key).model

        warmup_model, _ = load_with_hyperparameters(
            warmup_model_path, setup_func=_resume_setup_func
        )

    if warmup_model is None and args.warmup_model is not None:
        from jax_cookbook import load as fbx_load

        logger.info("Loading pre-trained warm-start model from %s", args.warmup_model)

        def _model_setup_func(key=jr.PRNGKey(0), **stored_hps):
            """Reconstruct a model from stored hyperparameters."""
            # The stored hps come from the standard (non-adversarial) trainer,
            # so use its build_hps function from the rlrmp.train library.
            from rlrmp.train.standard import build_hps as build_hps_standard

            stored_hps.pop("git", None)
            stored_hps.pop("output_dir", None)
            stored_args = SimpleNamespace(**stored_hps)
            stored_hps_obj = build_hps_standard(stored_args)
            return setup_task_model_pair(stored_hps_obj, key=key).model

        warmup_model = fbx_load(args.warmup_model, setup_func=_model_setup_func)
        logger.info("Loaded warm-start model (skipping phase 1).")

    if warmup_model is None:
        logger.info(
            "Phase 1: warm-start for %d batches (controller_lr=%g)",
            args.n_warmup_batches,
            args.controller_lr,
        )

        warmup_schedule = make_delayed_cosine_schedule(
            args.controller_lr,
            constant_steps=0,
            total_steps=args.n_warmup_batches,
        )
        warmup_optimizer = optax.inject_hyperparams(partial(optax.adamw, weight_decay=0.0))(
            learning_rate=warmup_schedule
        )

        chkpt_dir = output_dir / "checkpoints_warmup"
        chkpt_dir.mkdir(parents=True, exist_ok=True)
        warmup_trainer = TaskTrainer(
            optimizer=warmup_optimizer,
            checkpointing=True,
            chkpt_dir=chkpt_dir,
        )

        # Grep-friendly per-batch progress for remote monitoring. feedbax fires
        # these no-arg callbacks host-side (outside the JIT step), so they add
        # no per-step device->host sync. The loss is not visible to a no-arg
        # callback, so warmup lines report batch index + elapsed only.
        warmup_batch_callbacks = make_batch_log_callbacks(
            "warmup", args.n_warmup_batches, logger=logger
        )
        warmup_model, warmup_history = train_pair(
            warmup_trainer,
            pair,
            n_batches=args.n_warmup_batches,
            key=key_warmup,
            ensembled=True,
            loss_func=loss_func,
            where_train=where_train,
            batch_size=hps.batch_size,
            log_step=max(1, args.n_warmup_batches // 20),
            batch_callbacks=warmup_batch_callbacks,
        )
        logger.info("Warm-start complete.")

    # Enable jax.checkpoint on the model's scan body to trade compute for VRAM.
    # Must be done before flattening the model for the adversarial phase.
    # The checkpoint field is static so it doesn't affect array shapes.
    if args.checkpoint:
        try:
            object.__setattr__(warmup_model, "checkpoint", True)
            logger.info("Enabled jax.checkpoint on model scan body")
        except Exception:
            logger.warning("Could not enable checkpoint on model — flag has no effect")

    # Save warm-started model as a compatibility materialization (skip if we
    # loaded it from this path via --resume). The durable authority for the
    # warm-start controller is the after_warmup custody transaction written
    # below; this output_dir file is a convenience for downstream loaders and is
    # atomically staged (tmp-rooted → guard-classified ephemeral). Issue 7e71950.
    tmp_warmup_model_path = warmup_model_path.with_name("tmp_" + warmup_model_path.name)
    fbx_save(tmp_warmup_model_path, warmup_model, hyperparameters=config_dict)
    os.replace(tmp_warmup_model_path, warmup_model_path)
    logger.info("Saved warm-start model to %s", warmup_model_path)

    use_linear_dynamics = args.adversary_type == "linear_dynamics"

    # -----------------------------------------------------------------------
    # Phase 2 — adversarial training
    # -----------------------------------------------------------------------
    n_adversaries = args.n_adversaries
    n_reps = hps.model.n_replicates
    logger.info(
        "Phase 2: adversarial training for %d batches "
        "(adversary_type=%s, n_replicates=%d vmapped, n_adversaries=%d, "
        "n_adversary_steps=%d, adversary_lr=%g, controller_lr=%g, "
        "loss_update_enabled=%s, loss_update_ratio=%g)",
        args.n_adversary_batches,
        args.adversary_type,
        n_reps,
        n_adversaries,
        args.n_adversary_steps,
        args.adversary_lr,
        args.controller_lr,
        args.loss_update_enabled,
        args.loss_update_ratio,
    )

    # Create adversary population (K independent adversaries with different seeds).
    # Each adversary is vmapped across n_reps replicates so every replicate gets
    # its own independent adversary parameters.
    n_timesteps = hps.task.n_steps - 1  # feedbax uses n_steps-1 as the sim length

    def _make_adversary_population(n_adversaries: int) -> list:
        """Create K adversaries, each vmapped over n_reps replicates.

        Dispatches on ``args.adversary_type``:
        - ``gaussian_bump``: ``GaussianBumpAdversary`` (force-profile).
        - ``linear_dynamics``: ``LinearDynamicsAdversary`` (ΔA·x).
        """
        pop = []
        for i in range(n_adversaries):
            # Each replicate within adversary i gets a unique key
            rep_keys = jr.split(jr.PRNGKey(7 + i), n_reps)
            if args.adversary_type == "gaussian_bump":
                adv_vmapped = eqx.filter_vmap(
                    lambda k: GaussianBumpAdversary(
                        n_bumps=args.n_bumps,
                        n_timesteps=n_timesteps,
                        n_force_dims=2,
                        force_max=args.force_max,
                        dt=hps.dt,
                        key=k,
                    )
                )(rep_keys)
            elif args.adversary_type == "linear_dynamics":
                adv_vmapped = eqx.filter_vmap(
                    lambda k: LinearDynamicsAdversary(
                        n_state=4,
                        n_dim=2,
                        eta_max=args.linear_dynamics_eta_max,
                        n_inner_steps=args.n_adversary_steps,
                        learning_rate=args.linear_dynamics_lr,
                        key=k,
                    )
                )(rep_keys)
            else:
                raise ValueError(f"Unknown adversary_type: {args.adversary_type}")
            pop.append(adv_vmapped)
        return pop

    adversaries = _make_adversary_population(n_adversaries)

    # Adversary optimizers (one per adversary population member).
    # Init on a single-replicate adversary, then stack n_reps copies so ALL
    # state arrays (including step counters) carry a leading (n_reps,) axis.
    # The linear_dynamics adversary uses its own learning_rate (passed via
    # CLI), distinct from the GaussianBump adversary's --adversary-lr.
    adv_lr = (
        args.linear_dynamics_lr if args.adversary_type == "linear_dynamics" else args.adversary_lr
    )
    adversary_optimizer = optax.adam(adv_lr)

    def _init_vmapped_opt_state(vmapped_adv):
        """Init optimizer on one replicate's adversary, stack for all reps."""
        single_adv = jt.map(
            lambda x: x[0] if (eqx.is_array(x) and x.ndim > 0) else x,
            vmapped_adv,
            is_leaf=eqx.is_array,
        )
        single_st = adversary_optimizer.init(eqx.filter(single_adv, eqx.is_array))
        return jt.map(
            lambda x: jnp.stack([x] * n_reps) if eqx.is_array(x) else x,
            single_st,
            is_leaf=eqx.is_array,
        )

    adv_opt_states = [_init_vmapped_opt_state(adv) for adv in adversaries]

    # Controller optimizer (constant LR for the adversarial phase).
    # We train only the recurrent net weights (hidden + readout), same as TaskTrainer.
    ctrl_optimizer = optax.adamw(args.controller_lr, weight_decay=0.0)

    # The model from train_pair with ensembled=True has a leading replicate axis
    # on MOST array leaves: shape (n_reps, ...). Some arrays are shared across
    # replicates (no leading axis). We separate these for vmapping.
    #
    # Pre-flatten strategy (Bug: d6cc111): feedbax models store JAX arrays as
    # static PyTree metadata, so passing the model directly to filter_jit causes
    # recompilation after every update. We pass only flat arrays (all dynamic)
    # and close over the treedef (fixed).
    #
    # For vmapping: we split arrays into "per-replicate" (leading n_reps axis)
    # and "shared" (no replicate axis). The vmapped function receives per-rep
    # arrays; shared arrays are closed over. Both are recombined inside the
    # function via treedef_model.unflatten.

    def _has_rep_axis(x):
        return eqx.is_array(x) and x.ndim > 0 and x.shape[0] == n_reps

    # Extract single replicate for treedef (structure is same across replicates)
    single_rep_model = jt.map(
        lambda x: x[0] if _has_rep_axis(x) else x,
        warmup_model,
        is_leaf=eqx.is_array,
    )
    flat_single_rep, treedef_model = jtu.tree_flatten(single_rep_model)

    # Build masks: which flat leaves are per-replicate vs shared
    flat_ensembled = jtu.tree_flatten(warmup_model)[0]
    is_per_rep = [_has_rep_axis(x) for x in flat_ensembled]

    # Split into per-replicate arrays (vmapped) and shared arrays (closed over)
    per_rep_arrays = [x for x, pr in zip(flat_ensembled, is_per_rep) if pr]
    shared_leaves = [x for x, pr in zip(flat_ensembled, is_per_rep) if not pr]

    def _recombine_flat(per_rep_list, shared_list):
        """Recombine per-rep and shared leaves into a full flat list."""
        result = []
        pr_idx, sh_idx = 0, 0
        for is_pr in is_per_rep:
            if is_pr:
                result.append(per_rep_list[pr_idx])
                pr_idx += 1
            else:
                result.append(shared_list[sh_idx])
                sh_idx += 1
        return result

    def _split_flat(flat_list):
        """Split a full flat list into per-rep and shared portions."""
        per_rep = [x for x, pr in zip(flat_list, is_per_rep) if pr]
        shared = [x for x, pr in zip(flat_list, is_per_rep) if not pr]
        return per_rep, shared

    # The "flat_model" for the training loop is just the per-replicate arrays.
    # Shared arrays are closed over inside the vmapped functions.
    flat_model = per_rep_arrays

    # Also keep the full ensembled treedef for final model reconstruction
    _, treedef_ensembled = jtu.tree_flatten(warmup_model)

    # Initialise controller optimizer state on a single replicate, then stack
    # n_reps copies so ALL state arrays (including step counters) carry a leading
    # (n_reps,) axis for the vmapped training loop.
    single_rep_ctrl_state = ctrl_optimizer.init(
        eqx.filter(_get_trainable(single_rep_model), eqx.is_array)
    )
    ctrl_opt_state = jt.map(
        lambda x: jnp.stack([x] * n_reps) if eqx.is_array(x) else x,
        single_rep_ctrl_state,
        is_leaf=eqx.is_array,
    )
    adv_checkpoint_dir = output_dir / _CHECKPOINT_DIR_NAME
    legacy_adv_checkpoint_path = adv_checkpoint_dir / _CHECKPOINT_SUBDIR
    if not (
        args.resume
        and (has_custody_checkpoint(adv_checkpoint_dir) or legacy_adv_checkpoint_path.exists())
    ):
        _write_warmup_boundary_checkpoint(
            adv_checkpoint_dir,
            training_spec=feedbax_spec,
            model=warmup_model,
            adversaries=adversaries,
            adv_opt_states=adv_opt_states,
            ctrl_opt_state=ctrl_opt_state,
            rng_key=key_adv,
            warmup_history=warmup_history,
        )

    # ---------------------------------------------------------------------------
    # JIT-compiled training steps — defined as closures over task, loss_func,
    # ctrl_optimizer, and treedef_model, all of which are fixed for the entire
    # adversarial phase. The model is passed as a flat list of arrays (flat_model)
    # so that eqx.filter_jit sees only dynamic leaves in its argument, never
    # static metadata that changes with each update. Bug: d6cc111
    #
    # All functions operate on a SINGLE replicate internally. They are wrapped
    # with eqx.filter_vmap at the call site so each replicate trains its own
    # model + adversary independently, sharing only trial_specs and PRNG keys.
    # ---------------------------------------------------------------------------

    # Adversarial phase batch size (may differ from warmup to reduce XLA compile time)
    adv_batch_size = args.adv_batch_size if args.adv_batch_size is not None else hps.batch_size
    logger.info("Adversarial phase batch size: %d", adv_batch_size)

    n_adversary_steps = args.n_adversary_steps

    def _unflatten_model(per_rep_flat):
        """Reconstruct a single-replicate model from per-rep arrays + shared leaves.

        Merges the dynamic per-replicate arrays with the static shared leaves
        (closed over), then unflattens using treedef_model.
        """
        full_flat = _recombine_flat(per_rep_flat, shared_leaves)
        return jtu.tree_unflatten(treedef_model, full_flat)

    def _reflatten_model(model):
        """Extract only the per-replicate arrays from a single-replicate model."""
        full_flat = jtu.tree_flatten(model)[0]
        per_rep, _ = _split_flat(full_flat)
        return per_rep

    def _single_rep_loss_and_force_grad(per_rep_flat, adversary, trial_specs, keys):
        """Compute loss and gradient w.r.t. force array for a single replicate.

        Args:
            per_rep_flat: Per-replicate model arrays for ONE replicate.
            adversary: GaussianBumpAdversary for this replicate.
            trial_specs: Batched trial specifications (shared across replicates).
            keys: Per-trial PRNG keys, shape (batch_size,).

        Returns:
            Tuple of (loss_scalar, dL_dforces) where dL_dforces has shape
            (batch_size, T, d).
        """
        model = _unflatten_model(per_rep_flat)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model,
            is_leaf=eqx.is_array,
        )

        force_profile = adversary()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (adv_batch_size, *force_profile.shape))

        def _loss_fn(f):
            ts = _inject_adversary_forces(trial_specs, f)
            return _compute_loss(model_sg, ts, keys)

        return jax.value_and_grad(_loss_fn)(forces)

    def _single_rep_controller_step(per_rep_flat, ctrl_opt_st, adversary, trial_specs, keys):
        """Single gradient-descent step on the controller for one replicate.

        Args:
            per_rep_flat: Per-replicate model arrays for ONE replicate.
            ctrl_opt_st: Controller optimizer state for this replicate.
            adversary: GaussianBumpAdversary for this replicate.
            trial_specs: Batched trial specifications (shared across replicates).
            keys: Per-trial PRNG keys, shape (batch_size,).

        Returns:
            Tuple of (per_rep_flat_updated, updated_opt_state, loss_scalar).
        """
        model = _unflatten_model(per_rep_flat)

        force_profile = adversary()  # (T, d)
        forces = jnp.broadcast_to(force_profile, (adv_batch_size, *force_profile.shape))
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

        def _ctrl_loss(m):
            return _compute_loss(m, adv_trial_specs, keys)

        loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)

        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, new_opt_st = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            _trainable_where(model),
            model,
            updated_trainable,
        )
        return _reflatten_model(new_model), new_opt_st, loss_val

    # ---------------------------------------------------------------------------
    # Fused adversary batch — single JIT call replaces K×2 + 1 round-trips.
    # Uses lax.fori_loop for the inner adversary ascent steps, then performs a
    # single controller descent step. Closes over the same fixed objects as the
    # decomposed functions above. Bug: d6cc111
    # ---------------------------------------------------------------------------

    def _single_rep_fused_batch(
        per_rep_flat, adversary, adv_opt_st, ctrl_opt_st, trial_specs, keys
    ):
        """Fused adversary inner loop + controller step for a single replicate.

        Args:
            per_rep_flat: Per-replicate model arrays for ONE replicate.
            adversary: GaussianBumpAdversary for this replicate.
            adv_opt_st: Adversary optimizer state for this replicate.
            ctrl_opt_st: Controller optimizer state for this replicate.
            trial_specs: Batched trial specifications (shared across replicates).
            keys: Per-trial PRNG keys, shape (batch_size,).

        Returns:
            Tuple of (per_rep_flat_new, adversary_new, adv_opt_st_new,
            ctrl_opt_st_new, adv_loss, ctrl_loss).
        """
        model = _unflatten_model(per_rep_flat)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model,
            is_leaf=eqx.is_array,
        )

        # --- Inner adversary loop (K ascent steps) via lax.fori_loop ---
        def _adv_body(i, carry):
            adv, opt_st, _last_loss = carry

            force_profile = adv()  # (T, d)
            forces = jnp.broadcast_to(force_profile, (adv_batch_size, *force_profile.shape))

            def _loss_fn(f):
                ts = _inject_adversary_forces(trial_specs, f)
                return _compute_loss(model_sg, ts, keys)

            loss_val, dL_dforces = jax.value_and_grad(_loss_fn)(forces)

            def _forces_fn(a):
                fp = a()
                return jnp.broadcast_to(fp, dL_dforces.shape)

            _, vjp_fn = jax.vjp(lambda a: eqx.filter(_forces_fn(a), eqx.is_array), adv)
            neg_dL = jt.map(lambda g: -g, dL_dforces)
            (param_grads,) = vjp_fn(neg_dL)

            updates, new_opt_st = adversary_optimizer.update(
                eqx.filter(param_grads, eqx.is_array),
                opt_st,
                eqx.filter(adv, eqx.is_array),
            )
            new_adv = eqx.apply_updates(adv, updates)
            return new_adv, new_opt_st, loss_val

        init_carry = (adversary, adv_opt_st, jnp.asarray(0.0))
        adversary_new, adv_opt_st_new, adv_loss = jax.lax.fori_loop(
            0, n_adversary_steps, _adv_body, init_carry
        )

        # --- Controller descent step (1 step) ---
        force_profile = adversary_new()
        forces = jnp.broadcast_to(force_profile, (adv_batch_size, *force_profile.shape))
        adv_trial_specs = _inject_adversary_forces(trial_specs, forces)

        def _ctrl_loss(m):
            return _compute_loss(m, adv_trial_specs, keys)

        ctrl_loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)

        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, ctrl_opt_st_new = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            _trainable_where(model),
            model,
            updated_trainable,
        )

        return (
            _reflatten_model(new_model),
            adversary_new,
            adv_opt_st_new,
            ctrl_opt_st_new,
            adv_loss,
            ctrl_loss_val,
        )

    def _single_rep_fused_batch_linear_dynamics(
        per_rep_flat,
        adversary,
        adv_opt_st,
        ctrl_opt_st,
        trial_specs,
        keys,
    ):
        """Fused inner-loop + controller step for ``LinearDynamicsAdversary``.

        Mirrors ``_single_rep_fused_batch`` but injects the adversary's
        ``ΔA`` matrix into trial_specs via ``_inject_adversary_delta_A``,
        and applies a Frobenius-ball projection after each PGD step. Bug:
        c723082.
        """
        model = _unflatten_model(per_rep_flat)
        model_sg = jt.map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
            model,
            is_leaf=eqx.is_array,
        )

        def _adv_body(i, carry):
            adv, opt_st, _last_loss = carry

            # Loss as a function of ``adv.delta_A`` directly.
            def _loss_fn(a):
                ts = _inject_adversary_delta_A(
                    trial_specs,
                    a.delta_A,
                    adv_batch_size,
                )
                return _compute_loss(model_sg, ts, keys)

            loss_val, grads = eqx.filter_value_and_grad(_loss_fn)(adv)
            # Negate for gradient ascent
            neg_grads = jt.map(
                lambda g: -g if eqx.is_array(g) else g,
                grads,
            )
            updates, new_opt_st = adversary_optimizer.update(
                eqx.filter(neg_grads, eqx.is_array),
                opt_st,
                eqx.filter(adv, eqx.is_array),
            )
            new_adv = eqx.apply_updates(adv, updates)
            # Project to Frobenius ball (||delta_A||_F ≤ eta_max)
            new_adv = new_adv.project()
            return new_adv, new_opt_st, loss_val

        init_carry = (adversary, adv_opt_st, jnp.asarray(0.0))
        adversary_new, adv_opt_st_new, adv_loss = jax.lax.fori_loop(
            0,
            n_adversary_steps,
            _adv_body,
            init_carry,
        )

        # --- Controller descent step (1 step) ---
        adv_trial_specs = _inject_adversary_delta_A(
            trial_specs,
            adversary_new.delta_A,
            adv_batch_size,
        )

        def _ctrl_loss(m):
            return _compute_loss(m, adv_trial_specs, keys)

        ctrl_loss_val, grads = eqx.filter_value_and_grad(_ctrl_loss)(model)
        trainable_grads = eqx.filter(_get_trainable(grads), eqx.is_array)
        updates, ctrl_opt_st_new = ctrl_optimizer.update(
            trainable_grads,
            ctrl_opt_st,
            eqx.filter(_get_trainable(model), eqx.is_array),
        )
        updated_trainable = eqx.apply_updates(_get_trainable(model), updates)
        new_model = eqx.tree_at(
            _trainable_where(model),
            model,
            updated_trainable,
        )

        return (
            _reflatten_model(new_model),
            adversary_new,
            adv_opt_st_new,
            ctrl_opt_st_new,
            adv_loss,
            ctrl_loss_val,
        )

    # ---------------------------------------------------------------------------
    # Vmapped + JIT wrappers: vmap over replicate axis (0) for model, adversary,
    # and optimizer states; trial_specs and keys are broadcast (shared).
    # ---------------------------------------------------------------------------

    @eqx.filter_jit
    def _vmapped_fused_batch(flat_model, adversary, adv_opt_st, ctrl_opt_st, trial_specs, keys):
        """Fused adversary batch vmapped over replicates.

        flat_model, adversary, adv_opt_st, ctrl_opt_st have leading (n_reps,)
        on array leaves and are vmapped. trial_specs and keys are shared
        across replicates (closed over via lambda).

        Args:
            flat_model: Flat list of arrays, each (n_reps, ...).
            adversary: Vmapped GaussianBumpAdversary (arrays have leading n_reps).
            adv_opt_st: Vmapped adversary optimizer state.
            ctrl_opt_st: Vmapped controller optimizer state.
            trial_specs: Batched trial specs (shared across replicates).
            keys: Per-trial PRNG keys (shared across replicates).

        Returns:
            Same as _single_rep_fused_batch, with leading (n_reps,) on arrays.
        """
        # Close over trial_specs/keys so they are NOT vmapped; only the
        # per-replicate state (model, adversary, opt states) is vmapped.
        return eqx.filter_vmap(
            lambda fm, adv, aos, cos: _single_rep_fused_batch(fm, adv, aos, cos, trial_specs, keys)
        )(flat_model, adversary, adv_opt_st, ctrl_opt_st)

    @eqx.filter_jit
    def _vmapped_fused_batch_linear_dynamics(
        flat_model,
        adversary,
        adv_opt_st,
        ctrl_opt_st,
        trial_specs,
        keys,
    ):
        """Linear-dynamics fused adversary batch vmapped over replicates."""
        return eqx.filter_vmap(
            lambda fm, adv, aos, cos: _single_rep_fused_batch_linear_dynamics(
                fm, adv, aos, cos, trial_specs, keys
            )
        )(flat_model, adversary, adv_opt_st, ctrl_opt_st)

    @eqx.filter_jit
    def _vmapped_loss_and_force_grad(flat_model, adversary, trial_specs, keys):
        """Loss and force grad vmapped over replicates."""
        return eqx.filter_vmap(
            lambda fm, adv: _single_rep_loss_and_force_grad(fm, adv, trial_specs, keys)
        )(flat_model, adversary)

    @eqx.filter_jit
    def _vmapped_controller_step(flat_model, ctrl_opt_st, adversary, trial_specs, keys):
        """Controller step vmapped over replicates."""
        return eqx.filter_vmap(
            lambda fm, cos, adv: _single_rep_controller_step(fm, cos, adv, trial_specs, keys)
        )(flat_model, ctrl_opt_st, adversary)

    @eqx.filter_jit
    def _vmapped_adversary_update(adversary, dL_dforces, adv_opt_st):
        """Adversary update vmapped over replicates (decomposed mode only)."""
        return eqx.filter_vmap(partial(_adversary_update, adversary_optimizer))(
            adversary, dL_dforces, adv_opt_st
        )

    # -----------------------------------------------------------------------
    # Resume from checkpoint (if requested)
    # -----------------------------------------------------------------------
    start_batch_idx = 0
    adv_losses = []
    ctrl_losses = []
    adv_indices = []  # track which adversary was active each batch

    if args.resume:
        ckpt_path = adv_checkpoint_dir / _CHECKPOINT_SUBDIR
        if has_custody_checkpoint(adv_checkpoint_dir) or ckpt_path.exists():
            logger.info("--resume: loading adversarial checkpoint from %s", adv_checkpoint_dir)
            loaded_checkpoint = _load_adversarial_checkpoint(
                adv_checkpoint_dir,
                warmup_model,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                treedef_ensembled,
                training_spec=(
                    feedbax_spec if has_custody_checkpoint(adv_checkpoint_dir) else None
                ),
            )
            if len(loaded_checkpoint) == 9:
                (
                    full_flat_loaded,
                    adversaries,
                    adv_opt_states,
                    ctrl_opt_state,
                    last_completed_batch,
                    adv_losses,
                    ctrl_losses,
                    adv_indices,
                    key_adv,
                ) = loaded_checkpoint
            else:
                (
                    full_flat_loaded,
                    adversaries,
                    adv_opt_states,
                    ctrl_opt_state,
                    last_completed_batch,
                    adv_losses,
                    ctrl_losses,
                    adv_indices,
                ) = loaded_checkpoint
            # Extract only the per-replicate arrays for the training loop
            flat_model, _ = _split_flat(full_flat_loaded)
            start_batch_idx = last_completed_batch + 1
            logger.info(
                "Resuming adversarial training from batch %d/%d",
                start_batch_idx,
                args.n_adversary_batches,
            )
        else:
            logger.warning(
                "--resume was set but no checkpoint found at %s — starting from scratch.",
                ckpt_path,
            )

    # Periodic logging interval for the detailed adversarial summary line.
    log_step = max(1, args.n_adversary_batches // 20)
    # Cadence for the grep-friendly BATCH progress line consumed by remote
    # monitors (every batch for smoke sizes, else every Nth batch).
    batch_log_step = batch_log_every(args.n_adversary_batches)
    adv_phase_start = time.monotonic()

    use_fused = args.fused
    if use_fused:
        logger.info(
            "Using FUSED adversary batch (single JIT call with lax.fori_loop "
            "for %d inner adversary steps + 1 controller step)",
            n_adversary_steps,
        )
    else:
        logger.info(
            "Using DECOMPOSED adversary batch (%d×2 + 1 = %d separate JIT calls per batch)",
            n_adversary_steps,
            2 * n_adversary_steps + 1,
        )

    for batch_idx in range(start_batch_idx, args.n_adversary_batches):
        batch_key, key_adv = jr.split(key_adv)
        trial_keys = jr.split(batch_key, adv_batch_size)

        # Sample trial specs with intervenor params (needed for SISU/scale values).
        # BatchInfo fields must be JAX arrays (not Python ints) so that
        # filter_jit on get_train_trial_with_intervenor_params treats them as
        # dynamic traced values.  Python ints are static in eqx.Module and
        # would cause recompilation every batch, leaking ~0.5 GB/min of host
        # memory from the accumulated compilation cache.  Bug: d6cc111
        batch_info = BatchInfo(
            size=jnp.int32(adv_batch_size),
            current=jnp.int32(batch_idx),
            total=jnp.int32(args.n_adversary_batches),
        )
        trial_specs = jax.vmap(
            lambda key: task.get_train_trial_with_intervenor_params(key, batch_info)
        )(trial_keys)

        # task.eval_trials calls int(timeline.n_steps) which fails on traced arrays.
        # All trials have the same n_steps=140 (the fixed trial length); materialize
        # it as a concrete Python int so the call succeeds inside filter_grad.
        trial_specs = eqx.tree_at(
            lambda ts: ts.timeline.n_steps,
            trial_specs,
            int(hps.task.n_steps),
        )

        # Select active adversary (deterministic rotation for equal usage).
        # Each adversary in the population is vmapped across n_reps replicates.
        adv_idx = batch_idx % n_adversaries
        adversary = adversaries[adv_idx]
        adv_opt_state = adv_opt_states[adv_idx]
        adv_indices.append(adv_idx)

        if use_fused:
            # --- Single fused JIT call: K adversary steps + 1 controller step ---
            # Vmapped over replicates: each replicate trains independently.
            # Dispatch on adversary type (Bug: c723082).
            if use_linear_dynamics:
                fused_call = _vmapped_fused_batch_linear_dynamics
            else:
                fused_call = _vmapped_fused_batch
            (
                flat_model,
                adversary,
                adv_opt_state,
                ctrl_opt_state,
                adv_loss_vals,
                ctrl_loss_vals,
            ) = fused_call(
                flat_model,
                adversary,
                adv_opt_state,
                ctrl_opt_state,
                trial_specs,
                trial_keys,
            )
        else:
            # --- Decomposed: K×2 + 1 separate JIT calls per batch ---
            # Vmapped over replicates. Decomposed mode is only supported for
            # the gaussian_bump adversary; linear_dynamics requires the fused
            # path (its inner step uses ``filter_value_and_grad`` over the
            # adversary directly, not a separate force-grad VJP).
            if use_linear_dynamics:
                raise ValueError(
                    "--no-fused is not supported with --adversary-type "
                    "linear_dynamics; pass --fused (default)."
                )
            adv_loss_vals = jnp.zeros(n_reps)
            for _ in range(args.n_adversary_steps):
                adv_loss_vals, dL_dforces = _vmapped_loss_and_force_grad(
                    flat_model,
                    adversary,
                    trial_specs,
                    trial_keys,
                )
                adversary, adv_opt_state = _vmapped_adversary_update(
                    adversary,
                    dL_dforces,
                    adv_opt_state,
                )

            # Controller update (1 descent step)
            flat_model, ctrl_opt_state, ctrl_loss_vals = _vmapped_controller_step(
                flat_model,
                ctrl_opt_state,
                adversary,
                trial_specs,
                trial_keys,
            )

        # Write back updated adversary and optimizer state
        adversaries[adv_idx] = adversary
        adv_opt_states[adv_idx] = adv_opt_state

        # Losses are (n_reps,) arrays; store per-replicate means for history
        adv_loss_mean = float(jnp.mean(adv_loss_vals))
        ctrl_loss_mean = float(jnp.mean(ctrl_loss_vals))
        adv_losses.append(adv_loss_mean)
        ctrl_losses.append(ctrl_loss_mean)

        # Grep-friendly BATCH progress line for remote monitors. Reuses the
        # loss scalars already synced to host above (no extra device->host
        # sync). `loss` reports the controller loss; `adv_loss` is appended as
        # an extra field.
        if should_log_batch(batch_idx, args.n_adversary_batches, every=batch_log_step):
            logger.info(
                format_batch_line(
                    "adversarial",
                    batch_idx,
                    args.n_adversary_batches,
                    loss=ctrl_loss_mean,
                    elapsed=time.monotonic() - adv_phase_start,
                    adv_loss=adv_loss_mean,
                )
            )

        if batch_idx % log_step == 0 or batch_idx == args.n_adversary_batches - 1:
            adv_label = f" [adv {adv_idx}]" if n_adversaries > 1 else ""
            ctrl_std = float(jnp.std(ctrl_loss_vals))
            adv_std = float(jnp.std(adv_loss_vals))
            logger.info(
                "Adversarial batch %d/%d%s — ctrl_loss=%.4g +/- %.4g, "
                "adv_loss=%.4g +/- %.4g  (n_reps=%d)",
                batch_idx,
                args.n_adversary_batches,
                adv_label,
                ctrl_loss_mean,
                ctrl_std,
                adv_loss_mean,
                adv_std,
                n_reps,
            )

        # Periodic checkpoint (save after batch_idx is complete, not before)
        checkpoint_every = args.checkpoint_every
        if checkpoint_every > 0 and (batch_idx + 1) % checkpoint_every == 0:
            logger.info(
                "Saving adversarial checkpoint at batch %d → %s",
                batch_idx,
                adv_checkpoint_dir / _CHECKPOINT_SUBDIR,
            )
            _save_adversarial_checkpoint(
                adv_checkpoint_dir,
                _recombine_flat(flat_model, shared_leaves),
                treedef_ensembled,
                adversaries,
                adv_opt_states,
                ctrl_opt_state,
                batch_idx,
                adv_losses,
                ctrl_losses,
                adv_indices,
                training_spec=feedbax_spec,
                rng_key=key_adv,
            )

    logger.info("Adversarial training complete.")

    # Reconstruct the ensembled model from per-replicate + shared arrays.
    # flat_model contains only per-replicate arrays (with leading n_reps axis);
    # shared_leaves are the arrays/values without the replicate axis.
    full_flat = _recombine_flat(flat_model, shared_leaves)
    adv_model = jtu.tree_unflatten(treedef_ensembled, full_flat)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Durable authority: one terminal (status="final") custody transaction that
    # content-addresses the run's final controller, adversary population, loss
    # curves, and warmup history, published through the feedbax custody
    # latest-pointer run record. The output_dir .eqx/.npz files written below
    # are compatibility materializations for downstream loaders — atomically
    # staged (tmp-rooted → guard-classified ephemeral), no longer the durable
    # record. Issue 7e71950.
    final_controller = adv_model if args.n_adversary_batches > 0 else warmup_model
    _write_final_minimax_custody_transaction(
        adv_checkpoint_dir,
        training_spec=feedbax_spec,
        model=final_controller,
        adversaries=adversaries,
        adv_opt_states=adv_opt_states,
        ctrl_opt_state=ctrl_opt_state,
        rng_key=key_adv,
        batch_idx=args.n_adversary_batches - 1,
        adv_losses=adv_losses,
        ctrl_losses=ctrl_losses,
        adv_indices=adv_indices,
        warmup_history=warmup_history,
    )
    logger.info("Wrote terminal minimax custody transaction under %s", adv_checkpoint_dir)

    # Final adversarially-trained model (ensembled: arrays have leading n_reps axis).
    # Bug: a517040 — skip when n_adversary_batches=0: the adversarial phase did not
    # run, and the saved PyTree's adversary state does not match the local skeleton
    # produced by `setup_task_model_pair` at load time, breaking deserialization.
    # Downstream loaders fall back to `warmup_model.eqx` (the correct final model).
    if args.n_adversary_batches > 0:
        final_model_path = output_dir / "adversarial_model.eqx"
        tmp_final_model_path = final_model_path.with_name("tmp_" + final_model_path.name)
        fbx_save(tmp_final_model_path, adv_model, hyperparameters=config_dict)
        os.replace(tmp_final_model_path, final_model_path)
        logger.info("Saved adversarial model (n_reps=%d ensembled) to %s", n_reps, final_model_path)
    else:
        logger.info(
            "Skipping adversarial_model.eqx save (n_adversary_batches=0); "
            "warmup_model.eqx is the canonical final model for this run."
        )

    # Training histories (warmup from TaskTrainer; adversarial phase as numpy arrays)
    if warmup_history is not None:
        warmup_history_path = output_dir / "warmup_history.eqx"
        tmp_warmup_history_path = warmup_history_path.with_name("tmp_" + warmup_history_path.name)
        fbx_save(tmp_warmup_history_path, warmup_history)
        os.replace(tmp_warmup_history_path, warmup_history_path)
    loss_data = {
        "ctrl_losses": np.array(ctrl_losses),
        "adv_losses": np.array(adv_losses),
        "adv_indices": np.array(adv_indices),
    }
    adversarial_losses_path = output_dir / "adversarial_losses.npz"
    tmp_adversarial_losses_path = adversarial_losses_path.with_name(
        "tmp_" + adversarial_losses_path.name
    )
    np.savez(tmp_adversarial_losses_path, **loss_data)
    os.replace(tmp_adversarial_losses_path, adversarial_losses_path)
    logger.info("Saved adversarial loss curves to %s", adversarial_losses_path)

    # Final adversary/adversaries (each is vmapped across n_reps replicates). The
    # single- vs multi-adversary if/else and the force-profile vs ΔA log dispatch
    # are the conditional emitter structure the write-surface branch matrix
    # captures; keep both legs materializing so a single toy run cannot certify
    # the whole surface.
    log_fn = (
        _log_linear_dynamics_adversary if use_linear_dynamics else _log_adversary_force_profiles
    )
    if n_adversaries == 1:
        # Single adversary population: materialize with original filename for
        # backward compat (the adversary is content-addressed in the terminal
        # custody transaction above).
        trained_adversary_path = output_dir / "trained_adversary.eqx"
        tmp_trained_adversary_path = trained_adversary_path.with_name(
            "tmp_" + trained_adversary_path.name
        )
        fbx_save(tmp_trained_adversary_path, adversaries[0])
        os.replace(tmp_trained_adversary_path, trained_adversary_path)
        logger.info(
            "Saved trained adversary (n_reps=%d) to %s",
            n_reps,
            trained_adversary_path,
        )
        log_fn(adversaries[0], output_dir, n_reps=n_reps)
    else:
        adv_dir = output_dir / "adversaries"
        adv_dir.mkdir(parents=True, exist_ok=True)
        for i, adv in enumerate(adversaries):
            adv_path = adv_dir / f"adversary_{i}.eqx"
            tmp_adv_path = adv_path.with_name("tmp_" + adv_path.name)
            fbx_save(tmp_adv_path, adv)
            os.replace(tmp_adv_path, adv_path)
            logger.info("Saved adversary %d to %s", i, adv_path)
            log_fn(adv, output_dir, suffix=f"_adv{i}", n_reps=n_reps)
        logger.info("Saved %d adversaries (each n_reps=%d) to %s", n_adversaries, n_reps, adv_dir)

    logger.info("All results saved to %s", output_dir)


def _log_adversary_force_profiles(
    adversary: GaussianBumpAdversary,
    output_dir: Path,
    suffix: str = "",
    n_reps: int = 1,
) -> None:
    """Log adversary force profiles (SISU-independent) to a numpy archive.

    When the adversary is vmapped across replicates, generates and saves
    per-replicate force profiles with shape (n_reps, T, 2).

    Args:
        adversary: Trained GaussianBumpAdversary (possibly vmapped with leading
            n_reps axis on array leaves).
        output_dir: Directory to write the archive into.
        suffix: Optional suffix for the output filename (e.g. "_adv0").
        n_reps: Number of replicates (for logging).
    """
    # Generate force profiles: vmapped adversary produces (n_reps, T, 2)
    forces = eqx.filter_vmap(lambda a: a())(adversary)  # (n_reps, T, 2)
    forces_np = np.array(forces)
    per_rep_norms = np.linalg.norm(forces_np.reshape(n_reps, -1), axis=-1)
    logger.info(
        "Adversary%s force profile norms: mean=%.4g +/- %.4g (n_reps=%d)",
        suffix,
        per_rep_norms.mean(),
        per_rep_norms.std(),
        n_reps,
    )

    # Derived diagnostic materialization; the source adversary is
    # content-addressed in the terminal custody transaction. Atomically staged
    # (tmp-rooted → guard-classified ephemeral). Issue 7e71950.
    filename = f"adversary_force_profiles{suffix}.npz"
    force_profiles_path = output_dir / filename
    tmp_force_profiles_path = force_profiles_path.with_name("tmp_" + force_profiles_path.name)
    np.savez(tmp_force_profiles_path, forces=forces_np)
    os.replace(tmp_force_profiles_path, force_profiles_path)
    logger.info("Saved adversary force profiles to %s", force_profiles_path)


def _log_linear_dynamics_adversary(
    adversary: LinearDynamicsAdversary,
    output_dir: Path,
    suffix: str = "",
    n_reps: int = 1,
) -> None:
    """Log per-replicate ``ΔA`` matrices to a numpy archive.

    Counterpart to ``_log_adversary_force_profiles`` for the
    ``LinearDynamicsAdversary`` flavour. Bug: c723082.

    Args:
        adversary: Trained ``LinearDynamicsAdversary`` (vmapped over
            replicates: ``delta_A`` has shape ``(n_reps, n_dim, n_state)``).
        output_dir: Directory to write the archive into.
        suffix: Optional filename suffix.
        n_reps: Number of replicates (for logging).
    """
    deltas = eqx.filter_vmap(lambda a: a.delta_A)(adversary)
    deltas_np = np.array(deltas)
    norms = np.linalg.norm(deltas_np.reshape(n_reps, -1), axis=-1)
    logger.info(
        "Adversary%s ΔA Frobenius norms: mean=%.4g +/- %.4g (n_reps=%d)",
        suffix,
        norms.mean(),
        norms.std(),
        n_reps,
    )
    # Derived diagnostic materialization; the source adversary is
    # content-addressed in the terminal custody transaction. Atomically staged
    # (tmp-rooted → guard-classified ephemeral). Issue 7e71950.
    filename = f"adversary_delta_A{suffix}.npz"
    delta_A_path = output_dir / filename
    tmp_delta_A_path = delta_A_path.with_name("tmp_" + delta_A_path.name)
    np.savez(tmp_delta_A_path, delta_A=deltas_np)
    os.replace(tmp_delta_A_path, delta_A_path)
    logger.info("Saved adversary ΔA matrices to %s", delta_A_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> SimpleNamespace:
    return minimax_config_namespace(legacy_cli_args_to_minimax_config(sys.argv[1:]))



def author_minimax_training_run_spec(args: SimpleNamespace) -> TrainingRunSpec:
    """Author, validate, and write the minimax TrainingRunSpec for a launch."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = Path(args.spec_dir) if args.spec_dir is not None else derive_spec_dir(output_dir)
    mkdir_p(spec_dir)
    spec_path = derive_spec_path(output_dir)
    mkdir_p(spec_path.parent)

    hps = build_hps(args)
    key_init = jr.split(jr.PRNGKey(args.seed), 3)[0]
    pair = setup_task_model_pair(hps, key=key_init)
    graph_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair.model)
    graph_path = write_graph_spec_bundle(graph_bundle, spec_dir)
    payload = build_minimax_training_run_spec(
        args.__dict__,
        graph_spec=graph_bundle.graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
        git=_get_git_metadata(),
        gpu_info=_collect_gpu_info(),
        feedbax_graph=graph_bundle.to_run_metadata(graph_spec_path=graph_path.name),
    )
    validate_minimax_run_spec(payload, spec_dir=spec_dir)
    with open(spec_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved validated minimax TrainingRunSpec recipe to %s", spec_path)
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def training_run_spec_from_argv(argv: list[str]) -> TrainingRunSpec:
    """Return the spec requested by CLI argv without exposing raw args to training."""

    if argv[:1] == ["--training-run-spec"]:
        if len(argv) != 2:
            raise ValueError("--training-run-spec expects exactly one path")
        return minimax_training_run_spec_from_file(argv[1])
    args = minimax_config_namespace(legacy_cli_args_to_minimax_config(argv))
    return author_minimax_training_run_spec(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    spec = training_run_spec_from_argv(sys.argv[1:])
    run_training(spec)
