from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from rlrmp.analysis.soft_lambda import (
    materialize_write_print,
    run_soft_lambda_materializer,
)
from rlrmp.io import load_python_module, write_csv_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = (
    "results/06a4dc8/scripts/materialize_canonical_soft_lambda_hvp.py",
    "results/093d949/scripts/materialize_soft_lambda_sweep.py",
    "results/1697bdc/scripts/materialize_critical_lambda_search.py",
    "results/3b850d6/scripts/materialize_closed_loop_policy_audit.py",
    "results/6cfa892/scripts/materialize_closed_loop_soft_lambda_redo.py",
    "results/7180984/scripts/materialize_direct_epsilon_soft_lambda_redo.py",
    "results/d469108/scripts/materialize_adam_soft_lambda_redo.py",
    "results/f3c5db9/scripts/materialize_frozen_adam_audit_tuning.py",
)
MAIN_HELPER_BY_PATH = {
    path: "run_soft_lambda_materializer"
    for path in SCRIPT_PATHS
    if not path.startswith("results/093d949/")
}
BANNED_CSV_WRAPPERS = {"write_csv", "write_sweep_csv", "write_trial_csv"}


def _load(path: str, suffix: str) -> Any:
    return load_python_module(
        REPO_ROOT / path,
        module_name=f"_soft_lambda_dedupe_{suffix}",
    )


def _call_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _common_payload() -> dict[str, Any]:
    return {
        "issue": "example",
        "hvp_source": {
            "path": "results/source.json",
            "schema_version": "rlrmp.soft_lambda_hvp.v1",
            "primary_continuity_summary": "lambda_star_p90",
        },
        "rows": [
            {
                "run_id": "open_loop_small",
                "beta_mapping": [
                    {
                        "beta": 1.4,
                        "role": "candidate",
                        "lambda_star_summary_value": 2.0,
                        "lambda": 3.92,
                        "lambda_source": "hvp_p90",
                    }
                ],
                "objective_classification_counts": {"selected": 1},
            }
        ],
    }


def test_write_csv_rows_projects_generator_with_stable_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.csv"
    rows = ({"ignored": index, "b": index + 1, "a": index} for index in range(2))

    write_csv_rows(path, rows, fieldnames=("a", "b"))

    assert path.read_text(encoding="utf-8") == "a,b\n0,1\n1,2\n"
    with pytest.raises(KeyError, match="missing"):
        write_csv_rows(tmp_path / "bad.csv", [{"present": 1}], fieldnames=("missing",))


def test_materialize_drivers_materialize_once_and_preserve_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[object] = []

    materialize_write_print(
        materialize=lambda: events.append("materialize") or {"value": 2},
        writers=(
            lambda payload: events.append(("first", payload["value"])),
            lambda payload: events.append(("second", payload["value"])),
        ),
        summarize=lambda payload: events.append("summarize") or payload["value"] + 1,
        printer=lambda summary: events.append(("print", summary)),
    )
    assert events == [
        "materialize",
        ("first", 2),
        ("second", 2),
        "summarize",
        ("print", 3),
    ]

    calls = 0

    def materialize(_args: argparse.Namespace) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"rows": [{"a": 1, "ignored": 2}]}

    status = run_soft_lambda_materializer(
        args=argparse.Namespace(
            output_json="out.json",
            output_csv="out.csv",
            output_md="out.md",
        ),
        repo_root=tmp_path,
        materialize=materialize,
        csv_rows=lambda payload: payload["rows"],
        csv_fields=("a",),
        render_markdown=lambda _payload: "rendered",
        marker="test",
        extra_summary=lambda _payload: {"extra": True},
    )
    assert status == 0
    assert calls == 1
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == {
        "rows": [{"a": 1, "ignored": 2}]
    }
    assert (tmp_path / "out.csv").read_text(encoding="utf-8") == "a\n1\n"
    assert "rendered" in (tmp_path / "out.md").read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["extra"] is True


def test_soft_lambda_scripts_cannot_reaccrete_csv_or_main_drivers() -> None:
    for relative_path in SCRIPT_PATHS:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert "csv" not in imported_modules, relative_path
        assert not (function_names & BANNED_CSV_WRAPPERS), relative_path
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "DictWriter"
            for node in ast.walk(tree)
        ), relative_path

        expected_helper = MAIN_HELPER_BY_PATH.get(relative_path)
        if expected_helper is not None:
            main = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main"
            )
            assert expected_helper in _call_names(main), relative_path


def test_direct_and_closed_markdown_use_shared_hvp_beta_contract() -> None:
    direct = _load(SCRIPT_PATHS[5], "direct")
    closed = _load(SCRIPT_PATHS[4], "closed")
    direct_payload = _common_payload()
    direct_payload["rows"][0]["sweep"] = [
        {
            "beta": 1.4,
            "finite_status": "finite",
            "selected_nonzero": True,
            "classification": "selected",
            "penalized_gain_over_zero": 1.0,
            "task_loss_gain": 2.0,
            "energy_mean": 3.0,
            "energy_penalty": 4.0,
            "selected_epsilon_norm_max": 5.0,
            "old_cap_ratio_max_sidecar": 6.0,
        }
    ]
    closed_payload = _common_payload()
    closed_row = {
        "run_id": "open_loop_small",
        "mechanism": "affine",
        "optimizer": "adam",
        "beta": 1.4,
        "finite_status": "finite",
        "gradient_status": "finite",
        "classification": "selected",
        "objective_level_success": True,
        "penalized_gain_over_zero": 1.0,
        "task_loss_gain": 2.0,
        "energy_mean": 3.0,
        "energy_penalty": 4.0,
        "selected_policy_norm_max": 5.0,
        "old_cap_ratio_max_sidecar": 6.0,
    }
    closed_payload["flat_rows"] = [closed_row]
    closed_payload["rows"][0]["best_by_mechanism_optimizer"] = {"affine/adam": closed_row}
    closed_payload["overall_interpretation"] = "Closed-loop interpretation."

    direct_markdown = direct.render_markdown(direct_payload)
    closed_markdown = closed.render_markdown(closed_payload)

    for markdown in (direct_markdown, closed_markdown):
        assert markdown.count("## Source contract") == 1
        assert markdown.count("## HVP/p90 beta mapping") == 1
        assert "| `open_loop_small` | 1.4 | candidate | 2 | 3.92 | hvp_p90 |" in markdown
        assert "| `open_loop_small` | `selected`: 1 |" in markdown
    assert "## Direct-epsilon objective rows" in direct_markdown
    assert "## Best objective rows" in closed_markdown
    assert "Closed-loop interpretation." in closed_markdown
