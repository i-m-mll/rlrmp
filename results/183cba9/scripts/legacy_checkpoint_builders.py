"""Issue 183cba9 helpers for legacy C&S checkpoint manifest/adoption work.

These helpers are intentionally issue-local.  The LeafManifest dump entrypoint
is imported inside a temporary checkout at the legacy producing commit, while
the resume transform is imported from the current rlrmp checkout during
Feedbax checkpoint adoption.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp


def cs_nominal_gru_model_optimizer(spec_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return legacy model/optimizer templates for ``cs_nominal_gru`` checkpoints."""

    legacy_src = Path.cwd() / "src"
    if legacy_src.is_dir() and str(legacy_src) not in sys.path:
        sys.path.insert(0, str(legacy_src))
    _install_legacy_import_shims()

    from rlrmp.train.cs_nominal_gru import (
        _args_values_from_run_spec,
        _build_trainer,
        _initial_training_state,
        _where_train,
        build_hps,
        build_parser,
        setup_task_model_pair,
    )

    import jax.random as jr

    parser = build_parser()
    args = parser.parse_args([])
    for key, value in _args_values_from_run_spec(dict(spec_payload)).items():
        setattr(args, key, value)
    hps = build_hps(args)
    key_init, key_train, _key_adversary = jr.split(jr.PRNGKey(int(args.seed)), 3)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = _build_trainer(hps)
    state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=_where_train()[0],
        key=key_train,
    )
    return {"model": state.model, "optimizer": state.optimizer_state}


def _install_legacy_import_shims() -> None:
    persistence = types.ModuleType("feedbax.persistence")
    database = types.ModuleType("feedbax.persistence.database")

    class ModelRecord:
        pass

    def _unavailable(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("legacy manifest builder does not support database operations")

    database.ModelRecord = ModelRecord
    database.db_session = _unavailable
    database.get_db_session = _unavailable
    database.get_record = _unavailable
    database.save_model_and_add_record = _unavailable
    persistence.database = database
    sys.modules.setdefault("feedbax.persistence", persistence)
    sys.modules.setdefault("feedbax.persistence.database", database)


def adaptive_epsilon_adoption_resume_transform(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize adopted legacy trees into the current adaptive executor slot ABI."""

    from rlrmp.runtime.checkpoint_custody import serialize_pytree_slot
    from rlrmp.train.adaptive_epsilon_native import SerializedPyTreeSlot
    from rlrmp.train.cs_nominal_gru import _resize_optimizer_diagnostics_for_batches
    from rlrmp.train.executor.slots import (
        DAMAGE_METRIC,
        EPSILON_SCALE,
        MODEL,
        OPTIMIZER,
        TRAIN_LOSS,
    )

    payload = dict(slots)
    n_batches = _target_n_batches()
    if OPTIMIZER in payload:
        payload[OPTIMIZER] = _resize_optimizer_diagnostics_for_batches(
            payload[OPTIMIZER],
            n_batches,
        )
    if MODEL in payload and not isinstance(payload[MODEL], SerializedPyTreeSlot):
        payload[MODEL] = SerializedPyTreeSlot(serialize_pytree_slot(payload[MODEL]))
    if OPTIMIZER in payload and not isinstance(payload[OPTIMIZER], SerializedPyTreeSlot):
        payload[OPTIMIZER] = SerializedPyTreeSlot(serialize_pytree_slot(payload[OPTIMIZER]))
    payload[TRAIN_LOSS] = 0.0
    payload[DAMAGE_METRIC] = 0.0
    payload[EPSILON_SCALE] = 0.0
    return payload


def _target_n_batches() -> int:
    path = Path(__file__).resolve().parents[2] / "notes" / "adoption_context.json"
    if not path.is_file():
        return 12500
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("target_n_train_batches", 12500))


def leaf_summary(value: Any) -> dict[str, Any]:
    """Return a compact structural summary for local verification output."""

    arrays = []
    for path, leaf in __import__("jax").tree.leaves_with_path(value):
        if not eqx.is_array(leaf):
            continue
        arrays.append(
            {
                "path": "/" + "/".join(str(getattr(k, "name", getattr(k, "idx", k))) for k in path),
                "shape": tuple(int(dim) for dim in jnp.asarray(leaf).shape),
                "dtype": str(jnp.asarray(leaf).dtype),
            }
        )
    return {"array_count": len(arrays), "arrays": arrays[:8]}
