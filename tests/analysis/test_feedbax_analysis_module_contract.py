"""Contract checks for Feedbax-discoverable rlrmp analysis modules."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest
from feedbax.plugins.registry import ExperimentRegistry

import rlrmp


REQUIRED_ANALYSIS_MODULE_ATTRIBUTES = (
    "ANALYSES",
    "eval_fn",
    "setup_eval_tasks_and_models",
)


def test_registered_parts_exclude_removed_part1_and_keep_live_parts() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    metadata = registry.get_package_metadata("rlrmp")

    assert "part1" not in metadata.parts
    assert set(metadata.parts) == {"part2", "part3"}


def _registered_analysis_module_specs() -> list[tuple[str, Path]]:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    metadata = registry.get_package_metadata("rlrmp")
    root_name = f"{metadata.package_module.__name__}.{metadata.analysis_module_root}"

    module_specs: list[tuple[str, Path]] = []
    for part in metadata.parts:
        part_package = importlib.import_module(f"{root_name}.{part}")
        for info in pkgutil.iter_modules(part_package.__path__, prefix=f"{part_package.__name__}."):
            module_name = info.name
            short_name = module_name.rsplit(".", maxsplit=1)[-1]
            if info.ispkg or short_name.startswith("_"):
                continue

            spec = info.module_finder.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise AssertionError(
                    f"Could not resolve source path for analysis module {module_name}"
                )
            module_specs.append((module_name, Path(spec.origin)))

    return module_specs


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)

    return names


@pytest.mark.parametrize(
    ("module_name", "module_path"),
    _registered_analysis_module_specs(),
    ids=lambda value: value if isinstance(value, str) else value.name,
)
def test_registered_analysis_modules_define_feedbax_contract_attributes(
    module_name: str,
    module_path: Path,
) -> None:
    defined_names = _top_level_names(module_path)
    missing = [
        attribute
        for attribute in REQUIRED_ANALYSIS_MODULE_ATTRIBUTES
        if attribute not in defined_names
    ]

    assert missing == [], (
        f"{module_name} is registered under rlrmp's Feedbax analysis_module_root "
        f"but does not define required analysis-module attribute(s): {', '.join(missing)}. "
        "Define ANALYSES, eval_fn, and setup_eval_tasks_and_models at module top level."
    )
