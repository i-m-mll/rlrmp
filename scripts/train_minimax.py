"""Thin CLI adapter for the RLRMP minimax native training executor."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from feedbax import prepare_trial
from feedbax.contracts.training import TrainingRunSpec
from feedbax.objectives.streaming import make_streaming_loss_fn
from feedbax.runtime.iteration import run_component
from feedbax.training.checkpoint_custody import CheckpointCompatibilityError

from rlrmp.intervention_compat import LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
from rlrmp.model.feedbax_graph import (
    build_runtime_rlrmp_feedbax_graph_bundle,
    write_graph_spec_bundle,
)
from rlrmp.model.trainable import staged_network_trainable_parts
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path
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
from rlrmp.runtime.jax_config import assert_jax_x64_disabled
from rlrmp.train.minimax import (
    build_hps,
    build_minimax_training_run_spec,
    execute_minimax_training_run_spec_native,
    legacy_cli_args_to_minimax_config,
    minimax_config_namespace,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
)
from rlrmp.train.resume_control import emit_launch_continuation, resolve_launch_continuation
from rlrmp.train.task_model import setup_task_model_pair

logger = logging.getLogger(__name__)
_CHECKPOINT_SUBDIR = "checkpoint_latest"


def derive_spec_dir(output_dir: Path) -> Path:
    """Derive the tracked run sidecar/spec directory from the artifact directory."""

    out = Path(output_dir).resolve()
    artifact_root = (REPO_ROOT / "_artifacts").resolve()
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        return out.parent / (out.name + "_spec")


def derive_spec_path(output_dir: Path) -> Path:
    """Derive the canonical flat tracked run-recipe file from the artifact dir."""

    sidecar_dir = derive_spec_dir(output_dir)
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = sidecar_dir.resolve().relative_to(spec_root)
    except ValueError:
        return sidecar_dir.parent / (sidecar_dir.name + ".json")
    parts = rel.parts
    if len(parts) == 3 and parts[1] == "runs":
        return run_spec_path(parts[0], parts[2], for_write=True)
    return sidecar_dir.parent / (sidecar_dir.name + ".json")


def _get_git_metadata() -> dict[str, Any]:
    """Capture best-effort reproducibility metadata without blocking launch."""

    meta: dict[str, Any] = {}
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
    meta["jax_version"] = jax.__version__
    try:
        import feedbax

        meta["feedbax_version"] = getattr(feedbax, "__version__", "unknown")
    except ImportError:
        pass
    return meta


def _collect_gpu_info() -> dict[str, Any]:
    """Capture JAX-visible device info and optional ``nvidia-smi`` memory data."""

    info: dict[str, Any] = {}
    try:
        devices = jax.devices()
        info["device_kinds"] = [device.device_kind for device in devices]
        info["device_count"] = len(devices)
    except Exception as exc:  # pragma: no cover - defensive metadata only
        info["device_kinds"] = None
        info["device_count"] = 0
        info["jax_devices_error"] = str(exc)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        info["device_memory_gb_total"] = [
            float(value) / 1024.0 for value in result.stdout.strip().split("\n") if value
        ]
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as exc:
        info["device_memory_gb_total"] = None
        info["nvidia_smi_error"] = str(exc)
    return info


def _configure_jax_runtime(args: SimpleNamespace) -> None:
    """Configure JAX runtime options that must be set before first compile."""

    cache_dir = args.jax_cache_dir
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_path))
        logger.info("Using JAX compilation cache dir: %s", cache_path)
    if args.jax_explain_cache_misses:
        jax.config.update("jax_explain_cache_misses", True)


def parse_args() -> SimpleNamespace:
    """Parse legacy CLI flags into the governed minimax config namespace."""

    return minimax_config_namespace(legacy_cli_args_to_minimax_config(sys.argv[1:]))


def _save_adversarial_checkpoint(
    checkpoint_dir: Path,
    flat_model: list,
    treedef_model: Any,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state: Any,
    batch_idx: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
    *,
    training_spec: TrainingRunSpec | None = None,
    rng_key: Any = None,
) -> None:
    """Write one minimax adversarial checkpoint through custody plus materialization."""

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
    tmp_dir = checkpoint_dir / f"_{_CHECKPOINT_SUBDIR}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    try:
        eqx.tree_serialise_leaves(tmp_dir / "model.eqx", model)
        for index, adversary in enumerate(adversaries):
            eqx.tree_serialise_leaves(tmp_dir / f"adversary_{index}.eqx", adversary)
        eqx.tree_serialise_leaves(tmp_dir / "ctrl_opt_state.eqx", ctrl_opt_state)
        for index, opt_state in enumerate(adv_opt_states):
            eqx.tree_serialise_leaves(tmp_dir / f"adv_opt_state_{index}.eqx", opt_state)
        with open(tmp_dir / "meta.json", "w") as fh:
            json.dump(
                {
                    "batch_idx": batch_idx,
                    "n_adversaries": len(adversaries),
                    "adv_losses": adv_losses,
                    "ctrl_losses": ctrl_losses,
                    "adv_indices": adv_indices,
                },
                fh,
            )
        if target.exists():
            shutil.rmtree(target)
        tmp_dir.rename(target)
    except Exception:
        logger.exception("Checkpoint save failed; previous checkpoint is intact")
        raise


def _load_adversarial_checkpoint(
    checkpoint_dir: Path,
    model_template: Any,
    adversaries_template: list,
    adv_opt_states_template: list,
    ctrl_opt_state_template: Any,
    treedef_model: Any,
    *,
    training_spec: TrainingRunSpec | None = None,
) -> tuple[Any, ...]:
    """Load one minimax adversarial checkpoint from custody or materialization."""

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
    model = eqx.tree_deserialise_leaves(target / "model.eqx", model_template)
    flat_model = jtu.tree_flatten(model)[0]
    adversaries = [
        eqx.tree_deserialise_leaves(target / f"adversary_{index}.eqx", template)
        for index, template in enumerate(adversaries_template)
    ]
    ctrl_opt_state = eqx.tree_deserialise_leaves(
        target / "ctrl_opt_state.eqx",
        ctrl_opt_state_template,
    )
    adv_opt_states = [
        eqx.tree_deserialise_leaves(target / f"adv_opt_state_{index}.eqx", template)
        for index, template in enumerate(adv_opt_states_template)
    ]
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
    model: Any,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state: Any,
    rng_key: Any,
    warmup_history: Any,
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
    model: Any,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state: Any,
    rng_key: Any,
    batch_idx: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
    warmup_history: Any = None,
) -> None:
    """Write the terminal minimax custody transaction."""

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
    model: Any,
    adversaries: list,
    adv_opt_states: list,
    ctrl_opt_state: Any,
    rng_key: Any,
    batch_idx: int,
    active_member_index: int,
    adv_losses: list,
    ctrl_losses: list,
    adv_indices: list,
    training_spec: TrainingRunSpec,
    warmup_history: Any = None,
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
    model_template: Any,
    adversaries_template: list,
    adv_opt_states_template: list,
    ctrl_opt_state_template: Any,
) -> dict[str, object]:
    del model_template, adversaries_template, adv_opt_states_template, ctrl_opt_state_template
    return {
        "rng": jnp.asarray([0, 0], dtype=jnp.uint32),
        "active_batch_index": jnp.asarray(0, dtype=jnp.int32),
        "active_member_index": jnp.asarray(0, dtype=jnp.int32),
    }


def _adversary_population_member_ids(adversaries: list) -> dict[str, list[str]]:
    return {"adversary_population": [f"adversary_{index}" for index, _ in enumerate(adversaries)]}


def _get_trainable(model: Any) -> Any:
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


def _trainable_where(model: Any) -> Any:
    net = model.get_node("net")
    cls_name = type(net).__name__
    if cls_name == "AffineFeedbackController":
        if getattr(net, "feedforward", None) is not None:
            return lambda candidate: candidate.get_node_attrs("net", "gain", "feedforward")
        return lambda candidate: candidate.get_node_attrs("net", "gain")
    if cls_name == "LinearController":
        return lambda candidate: candidate.get_node_attrs("net", "K")
    if cls_name == "LinearTrackerController":
        return lambda candidate: candidate.get_node_attrs("net", "K", "u_ff")
    return lambda candidate: staged_network_trainable_parts(candidate.get_node("net"))


def _eval_trials_streaming(
    task: Any,
    model: Any,
    trial_specs: Any,
    keys: Any,
    loss_func: Any,
) -> Any:
    del task

    def eval_single(trial_spec: Any, key: Any) -> Any:
        key_run = jr.split(key, 2)[1]
        prepared = prepare_trial(model, trial_spec)
        prepared_inputs = _with_declarative_component_parameter_inputs(
            model,
            prepared.inputs,
        )
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

    return eqx.filter_vmap(eval_single)(trial_specs, keys).mean()


def _with_declarative_component_parameter_inputs(model: Any, inputs: Any) -> Any:
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
        primary_inputs = {
            key: value
            for key, value in inputs.items()
            if key != legacy_key and not key.startswith("task:")
        }
        return {"input": primary_inputs, declared_key: inputs[legacy_key]}
    return inputs


def author_minimax_training_run_spec(args: SimpleNamespace) -> TrainingRunSpec:
    """Author, validate, and write the minimax ``TrainingRunSpec`` for a launch."""

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


def run_training(training_run_spec: TrainingRunSpec | dict[str, Any]) -> Any:
    """Execute a minimax ``TrainingRunSpec`` through the native executor."""

    spec = (
        training_run_spec
        if isinstance(training_run_spec, TrainingRunSpec)
        else TrainingRunSpec.model_validate(training_run_spec)
    )
    args = minimax_config_namespace(minimax_training_run_spec_to_config(spec))
    assert_jax_x64_disabled("minimax training", allow_x64=bool(args.allow_x64))
    _configure_jax_runtime(args)
    checkpoint_root = Path(args.output_dir) / "checkpoints_adversarial"
    stop_target_batches = int(args.n_warmup_batches) + int(args.n_adversary_batches)
    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=bool(args.resume),
        allow_fresh_start=bool(args.allow_fresh_start),
        stop_target_batches=stop_target_batches,
        completed_batches_from_latest=lambda latest_path: _minimax_completed_batches_from_latest(
            latest_path,
            n_warmup_batches=int(args.n_warmup_batches),
            n_adversary_batches=int(args.n_adversary_batches),
        ),
    )
    emit_launch_continuation(continuation, logger=logger)
    return execute_minimax_training_run_spec_native(
        spec,
        run_id=Path(args.output_dir).name,
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        checkpoint_root=checkpoint_root,
        resume=continuation.resume,
        manifest_conflict_policy="reuse-identical",
    )


def _minimax_completed_batches_from_latest(
    latest_path: Path,
    *,
    n_warmup_batches: int,
    n_adversary_batches: int,
) -> int:
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    coordinate = payload.get("completed_coordinate")
    if not isinstance(coordinate, dict):
        raise ValueError(f"checkpoint latest pointer lacks completed_coordinate: {latest_path}")
    phase = coordinate.get("phase")
    barrier = coordinate.get("completed_barrier")
    global_step = int(coordinate.get("global_step", 0))
    total_batches = int(n_warmup_batches) + int(n_adversary_batches)
    if phase == "done":
        return total_batches
    if barrier == MINIMAX_WARMUP_BARRIER or phase == "warmup":
        return int(n_warmup_batches)
    if barrier == MINIMAX_ADVERSARIAL_BARRIER or phase == "adversarial":
        return min(total_batches, int(n_warmup_batches) + global_step)
    raise ValueError(f"unsupported minimax checkpoint coordinate in {latest_path}: {coordinate!r}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    spec = training_run_spec_from_argv(sys.argv[1:])
    run_training(spec)
