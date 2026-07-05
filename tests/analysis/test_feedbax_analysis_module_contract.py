"""Contract checks for Feedbax-discoverable rlrmp analysis recipes."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
from feedbax.analysis.specs import get_analysis_recipe, registered_analysis_types
from feedbax.analysis.validation import AnalysisRecipeProtocol, validate_analysis_recipe
from feedbax.plugins.registry import ExperimentRegistry

import rlrmp


pytestmark = pytest.mark.feedbax_contract


def test_registered_parts_exclude_removed_frozen_parts() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    metadata = registry.get_package_metadata("rlrmp")

    assert "part1" not in metadata.parts
    assert "part2" not in metadata.parts
    assert "part3" not in metadata.parts
    assert set(metadata.parts) == set()


def test_removed_frozen_analysis_parts_are_not_importable() -> None:
    assert importlib.util.find_spec("rlrmp.modules") is None


def _registered_rlrmp_analysis_recipes() -> list[tuple[str, AnalysisRecipeProtocol]]:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    recipe_types = sorted(
        analysis_type
        for analysis_type in registered_analysis_types()
        if analysis_type.startswith("rlrmp.")
    )
    assert recipe_types, "rlrmp registered zero Feedbax analysis recipes"
    return [
        (
            analysis_type,
            validate_analysis_recipe(analysis_type, get_analysis_recipe(analysis_type)),
        )
        for analysis_type in recipe_types
    ]


@pytest.mark.parametrize(
    ("analysis_type", "recipe"),
    _registered_rlrmp_analysis_recipes(),
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_registered_analysis_recipes_satisfy_feedbax_protocol(
    analysis_type: str,
    recipe: AnalysisRecipeProtocol,
) -> None:
    assert callable(recipe)
    assert validate_analysis_recipe(analysis_type, recipe) is recipe
