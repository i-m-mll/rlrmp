from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from rlrmp.io import load_named_python_module


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "results" / "08483d5" / "scripts"


@pytest.fixture
def script_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("_common")
    gru = load_named_python_module(
        "test_08483d5_gru_damage_sanity",
        SCRIPTS / "compute_gru_pgd_damage_sanity.py",
    )
    side = load_named_python_module(
        "test_08483d5_pgd_side_check",
        SCRIPTS / "compute_pgd_ofb_side_check.py",
    )
    spike = load_named_python_module(
        "test_08483d5_damage_spike",
        SCRIPTS / "compute_output_feedback_damage_spike_diagnostic.py",
    )
    return SimpleNamespace(common=common, gru=gru, side=side, spike=spike)


def _costs(total: tuple[float, float]) -> dict[str, jnp.ndarray]:
    return {
        "total": jnp.asarray(total),
        "stage_state": jnp.asarray(total) * 0.5,
        "control": jnp.asarray(total) * 0.25,
        "terminal": jnp.asarray(total) * 0.25,
    }


def test_canonical_damage_driver_runs_shared_pipeline_once(
    script_modules: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    common = script_modules.common
    run_spec = {"hps": {"model": {"n_replicates": 2}}}
    run = SimpleNamespace(run_spec=run_spec)
    trials = SimpleNamespace(
        inputs={"epsilon": jnp.zeros((2, 3, 6))},
        inits={"mechanics.vector": jnp.zeros((2, 6))},
        kind="clean",
    )
    task = SimpleNamespace(validation_trials=object())
    monkeypatch.setattr(common, "resolve_run_inputs", lambda **_kwargs: (run,))
    monkeypatch.setattr(common, "normalize_gru_hps", lambda value: value)
    monkeypatch.setattr(
        common,
        "dict_to_namespace",
        lambda *_args, **_kwargs: SimpleNamespace(model=SimpleNamespace(n_replicates=2)),
    )
    monkeypatch.setattr(
        common, "setup_task_model_pair", lambda *_args, **_kwargs: SimpleNamespace(task=task)
    )
    monkeypatch.setattr(
        common,
        "load_validation_selected_checkpoint_model",
        lambda **_kwargs: (object(), ("selection",)),
    )
    monkeypatch.setattr(common, "repeat_single_validation_trial", lambda *_args: trials)
    monkeypatch.setattr(common, "declared_active_radius", lambda *_args: 1.5)
    monkeypatch.setattr(common, "trial_target_position", lambda _trials: jnp.zeros((2, 2)))
    monkeypatch.setattr(common, "full_qrf_cost_context", lambda **_kwargs: {"context": True})
    monkeypatch.setattr(
        common,
        "with_epsilon_delta",
        lambda base, _delta: SimpleNamespace(**(vars(base) | {"kind": "adversarial"})),
    )
    calls: list[str] = []

    def fake_rollout_costs(*, trial_specs: Any, **_kwargs: Any) -> dict[str, jnp.ndarray]:
        calls.append(trial_specs.kind)
        return _costs((4.0, 6.0) if trial_specs.kind == "adversarial" else (1.0, 2.0))

    def fake_ascent(objective: Any, initial: Any, **_kwargs: Any) -> tuple[Any, Any, list]:
        delta = jnp.ones_like(initial) * 0.5
        value = objective(delta)
        return delta, value, [{"step": 0}, {"step": 1}, {"step": 2}]

    monkeypatch.setattr(common, "rollout_costs", fake_rollout_costs)
    monkeypatch.setattr(common, "projected_gradient_ascent", fake_ascent)
    result = common.compute_damage_row(
        experiment="example",
        run_id="row",
        label="Example row",
        repo_root=tmp_path,
        n_trials=2,
        n_steps=2,
        seed=7,
    )

    assert calls == ["adversarial", "clean", "adversarial"]
    assert result["checkpoint_selection"] == ("selection",)
    assert result["radius"] == 1.5
    assert result["clean_costs"]["total"]["mean"] == 1.5
    assert result["adversarial_costs"]["total"]["mean"] == 5.0
    assert result["damage"]["total"]["mean"] == 3.5


def _damage_computation(script_modules: SimpleNamespace) -> dict[str, Any]:
    contract = {
        "active_max_l2_radius_15cm": 0.4,
        "effective_l2_radius_15cm": 0.3,
        "gamma_factor": 1.4,
        "budget_source": "fixture",
    }
    run_spec = {
        "hps": {
            "hidden_type": "gru",
            "model": {
                "hidden_size": 8,
                "n_replicates": 2,
                "initial_hidden_encoder": False,
                "physical_state_dim": 6,
                "state_dim": 36,
                "no_integrator_state": True,
            },
            "loss": {"objective": "full_qrf"},
            "broad_epsilon_pgd_training": {
                "enabled": True,
                "level": "moderate",
                "mode": "output_feedback",
                "budget_contract": contract,
            },
        },
        "game_card": {"target_distance_m": 0.15},
    }
    trials = SimpleNamespace(
        inits={"mechanics.vector": np.zeros((2, 6))},
        targets={
            "mechanics.effector.pos": SimpleNamespace(value=np.zeros((2, 1, 2)))
        },
    )
    summary = {
        name: {"n": 2, "mean": mean, "std": 0.0, "min": mean, "max": mean}
        for name, mean in (
            ("total", 5.0),
            ("stage_state", 2.0),
            ("control", 1.0),
            ("terminal", 2.0),
        )
    }
    damage = {
        name: {"mean": value, "note": "mean(adversarial cost) - mean(clean cost)"}
        for name, value in (
            ("total", 2.0),
            ("stage_state", 1.0),
            ("control", 0.5),
            ("terminal", 0.5),
        )
    }

    class Selection:
        def to_json(self, *, repo_root: Path) -> dict[str, str]:
            return {"checkpoint": str(repo_root / "checkpoint.eqx")}

    return {
        "run": SimpleNamespace(
            run_spec_path=REPO_ROOT / "results" / "c92ebd8" / "runs" / "row.json",
            artifact_dir=REPO_ROOT / "_artifacts" / "c92ebd8" / "runs" / "row",
        ),
        "run_spec": run_spec,
        "checkpoint_selection": (Selection(),),
        "trial_specs": trials,
        "n_replicates": 2,
        "base_epsilon": np.zeros((2, 3, 6)),
        "best_delta": np.full((2, 3, 6), 0.1),
        "radius": 0.4,
        "history": [{"step": step} for step in range(11)],
        "clean_costs": summary,
        "adversarial_costs": summary,
        "damage": damage,
    }


def test_schema_adapters_preserve_both_historical_contracts(
    script_modules: SimpleNamespace,
) -> None:
    computation = _damage_computation(script_modules)
    gru = script_modules.gru.build_result(computation)
    row = {
        "label": "row label",
        "experiment": "c92ebd8",
        "run_id": "row",
    }
    side = script_modules.side.format_damage_row(row, computation)

    assert gru["schema_version"] == "rlrmp.08483d5_gru_damage_sanity.v1"
    assert gru["adversary"]["n_steps"] == 10
    assert gru["adversary"]["epsilon_dim"] == 6
    assert gru["adversary"]["budget_source"] == "fixture"
    assert gru["costs"]["paired_damage"] is computation["damage"]
    assert side["run_id"] == "row"
    assert side["adversary"]["budget_contract"]["gamma_factor"] == 1.4
    assert side["adversary"]["selected_epsilon"] == gru["adversary"]["selected_epsilon"]
    assert side["costs"]["paired_damage"] is computation["damage"]


def test_side_adapter_calls_canonical_driver(
    script_modules: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    computation = _damage_computation(script_modules)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        script_modules.side,
        "canonical_compute_damage_row",
        lambda **kwargs: calls.append(kwargs) or computation,
    )
    row = {"label": "row label", "experiment": "c92ebd8", "run_id": "row"}
    result = script_modules.side.compute_damage_row(row)

    assert result["run_id"] == "row"
    assert calls == [
        {
            "experiment": "c92ebd8",
            "run_id": "row",
            "label": "row label",
            "repo_root": REPO_ROOT,
            "n_trials": 64,
            "n_steps": 10,
            "seed": 42,
        }
    ]


def test_spike_loader_routes_canonical_io(
    script_modules: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = object()
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        script_modules.spike,
        "load_named_python_module",
        lambda name, path: calls.append((name, path)) or sentinel,
    )

    assert script_modules.spike._load_reference_module() is sentinel
    assert calls == [("damage_beta_curves_reference", script_modules.spike.SOURCE_PATH)]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(function: ast.FunctionDef) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_original_members_cannot_reaccrete_shared_orchestration() -> None:
    common = _function(SCRIPTS / "_common.py", "compute_damage_row")
    gru = _function(SCRIPTS / "compute_gru_pgd_damage_sanity.py", "main")
    side = _function(SCRIPTS / "compute_pgd_ofb_side_check.py", "compute_damage_row")
    spike = _function(
        SCRIPTS / "compute_output_feedback_damage_spike_diagnostic.py",
        "_load_reference_module",
    )
    forbidden = {
        "resolve_run_inputs",
        "load_validation_selected_checkpoint_model",
        "projected_gradient_ascent",
        "rollout_costs",
        "summarize_costs",
    }

    assert forbidden <= _calls(common)
    assert _calls(gru) & forbidden == set()
    assert _calls(side) & forbidden == set()
    assert "canonical_compute_damage_row" in _calls(gru)
    assert "canonical_compute_damage_row" in _calls(side)
    assert "load_named_python_module" in _calls(spike)
    assert gru.end_lineno - gru.lineno + 1 <= 16
    assert side.end_lineno - side.lineno + 1 <= 16
    assert spike.end_lineno - spike.lineno + 1 <= 2
    assert "importlib" not in (SCRIPTS / "compute_output_feedback_damage_spike_diagnostic.py").read_text()
