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
    class Figure:
        pass

    plotly = sys.modules.setdefault("plotly", types.ModuleType("plotly"))
    graph_objects = types.ModuleType("plotly.graph_objects")
    graph_objects.Figure = Figure
    sys.modules.setdefault("plotly.graph_objects", graph_objects)
    sys.modules.setdefault("plotly.graph_objs", graph_objects)
    plotly_io = types.ModuleType("plotly.io")
    plotly_io.templates = types.SimpleNamespace(default="plotly_white")
    sys.modules.setdefault("plotly.io", plotly_io)
    setattr(plotly, "graph_objects", graph_objects)
    setattr(plotly, "graph_objs", graph_objects)
    setattr(plotly, "io", plotly_io)

    feedbax_plot = types.ModuleType("feedbax.plot")
    feedbax_plot.utils = types.ModuleType("feedbax.plot.utils")
    feedbax_plot.utils.savefig = lambda *args, **kwargs: None
    sys.modules.setdefault("feedbax.plot", feedbax_plot)
    sys.modules.setdefault("feedbax.plot.utils", feedbax_plot.utils)

    analysis = types.ModuleType("feedbax.analysis")
    aligned = types.ModuleType("feedbax.analysis.aligned")
    aligned.get_aligned_vars = lambda *args, **kwargs: None
    aligned.get_reach_origins_directions = lambda *args, **kwargs: None
    state_utils = types.ModuleType("feedbax.analysis.state_utils")
    state_utils.get_pos_endpoints = lambda *args, **kwargs: None
    state_utils.vmap_eval_ensemble = lambda *args, **kwargs: None
    setup = types.ModuleType("feedbax.analysis.setup")
    setup.setup_models_only = lambda *args, **kwargs: None
    setup.setup_tasks_only = lambda *args, **kwargs: None
    sys.modules.setdefault("feedbax.analysis", analysis)
    sys.modules.setdefault("feedbax.analysis.aligned", aligned)
    sys.modules.setdefault("feedbax.analysis.state_utils", state_utils)
    sys.modules.setdefault("feedbax.analysis.setup", setup)

    persistence = types.ModuleType("feedbax.persistence")
    database = types.ModuleType("feedbax.persistence.database")

    class ModelRecord:
        pass

    class EvaluationRecord:
        pass

    def _unavailable(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("legacy manifest builder does not support database operations")

    database.MODEL_RECORD_BASE_ATTRS = ()
    database.EvaluationRecord = EvaluationRecord
    database.ModelRecord = ModelRecord
    database._cleanup_new_paths = _unavailable
    database.add_evaluation = _unavailable
    database.add_evaluation_figure = _unavailable
    database.check_model_files = _unavailable
    database.db_session = _unavailable
    database.get_db_session = _unavailable
    database.get_record = _unavailable
    database.load_tree_with_hps = _unavailable
    database.query_model_records = _unavailable
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
