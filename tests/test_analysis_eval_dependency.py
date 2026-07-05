"""Analysis-recipe evaluation-input dependency ratchet (issue 10bdaa4).

Registered rlrmp analysis recipes should consume evaluation manifests once the
analysis pipeline is decomposed into feedbax-native eval and analysis stages.
This gate freezes the current undeclared inventory in a shrink-only allowlist:
new registered analyses must declare their evaluation dependencies, and stale
exemptions must be removed when recipes are migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import re
import tomllib

import pytest
from feedbax.analysis.specs import get_analysis_recipe, registered_analysis_types
from feedbax.analysis.validation import AnalysisRecipeProtocol, validate_analysis_recipe
from feedbax.plugins.registry import ExperimentRegistry

import rlrmp


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "analysis-eval-dependency-allowlist.toml"


@dataclass(frozen=True)
class RegisteredAnalysisRecipe:
    """One registered rlrmp analysis recipe and its dependency declaration state."""

    analysis_type: str
    recipe_module: str
    recipe_name: str
    eval_dependencies: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.analysis_type, self.recipe_module, self.recipe_name)

    @property
    def declares_eval_dependency(self) -> bool:
        return bool(self.eval_dependencies)


def test_analysis_eval_dependency_scan_is_non_vacuous() -> None:
    recipes = _registered_rlrmp_analysis_recipes()

    assert recipes, "rlrmp registered zero Feedbax analysis recipes"
    analysis_types = {recipe.analysis_type for recipe in recipes}
    assert "rlrmp.gru_postrun" in analysis_types
    assert "rlrmp.standard_matrix" in analysis_types
    assert any(recipe.recipe_module == "rlrmp.analysis.declarative_materialization" for recipe in recipes)


def test_registered_analysis_eval_dependencies_match_allowlist() -> None:
    found = _undeclared_recipes_by_key(_registered_rlrmp_analysis_recipes())
    allowed = _allowlist_by_key(_load_allowlist())

    _assert_no_unlisted_undeclared_recipes(found, allowed)


def test_analysis_eval_dependency_allowlist_has_no_dead_entries() -> None:
    found = _undeclared_recipes_by_key(_registered_rlrmp_analysis_recipes())
    allowed = _allowlist_by_key(_load_allowlist())

    dead = sorted(set(allowed) - set(found))
    assert not dead, (
        "Analysis eval-dependency allowlist names recipe(s) that no longer need "
        f"an exemption: {dead}. Remove stale entries; shrinking the inventory is required."
    )


def test_analysis_eval_dependency_allowlist_entries_carry_owner_and_reason() -> None:
    issue_re = re.compile(r"^[0-9a-f]{7}$")
    entries = _load_allowlist().get("eval_dependency_exemptions", [])
    assert entries, "analysis eval-dependency allowlist declares zero exemptions"
    for entry in entries:
        assert issue_re.match(entry.get("owner", "")), (
            f"Allowlist entry {entry} is missing a 7-character owning issue"
        )
        assert isinstance(entry.get("reason"), str) and len(entry["reason"].strip()) >= 20, (
            f"Allowlist entry {entry} needs a brief reason"
        )


def test_analysis_eval_dependency_negative_canary_flags_unlisted_registered_analysis() -> None:
    found = {
        (
            "rlrmp.new_analysis",
            "rlrmp.analysis.new_module",
            "new_analysis_recipe",
        ): RegisteredAnalysisRecipe(
            analysis_type="rlrmp.new_analysis",
            recipe_module="rlrmp.analysis.new_module",
            recipe_name="new_analysis_recipe",
            eval_dependencies=(),
        )
    }
    allowed: dict[tuple[str, str, str], dict[str, str]] = {}

    with pytest.raises(AssertionError, match="Registered rlrmp analysis recipe"):
        _assert_no_unlisted_undeclared_recipes(found, allowed)


def _registered_rlrmp_analysis_recipes() -> list[RegisteredAnalysisRecipe]:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    recipe_types = sorted(
        analysis_type
        for analysis_type in registered_analysis_types()
        if analysis_type.startswith("rlrmp.")
    )
    return [
        _registered_recipe(
            analysis_type,
            validate_analysis_recipe(analysis_type, get_analysis_recipe(analysis_type)),
        )
        for analysis_type in recipe_types
    ]


def _registered_recipe(
    analysis_type: str,
    recipe: AnalysisRecipeProtocol,
) -> RegisteredAnalysisRecipe:
    return RegisteredAnalysisRecipe(
        analysis_type=analysis_type,
        recipe_module=recipe.__module__,
        recipe_name=recipe.__name__,
        eval_dependencies=_declared_eval_dependencies(analysis_type, recipe),
    )


def _declared_eval_dependencies(
    analysis_type: str,
    recipe: AnalysisRecipeProtocol,
) -> tuple[str, ...]:
    recipe_level = _normalise_dependencies(getattr(recipe, "EVAL_DEPENDENCIES", ()))
    if recipe_level:
        return recipe_level

    module = importlib.import_module(recipe.__module__)
    by_type = getattr(module, "EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE", {})
    if isinstance(by_type, dict) and analysis_type in by_type:
        return _normalise_dependencies(by_type[analysis_type])

    module_level = getattr(module, "EVAL_DEPENDENCIES", ())
    return _normalise_dependencies(module_level)


def _normalise_dependencies(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise AssertionError("EVAL_DEPENDENCIES must be a tuple/list of strings, not a string")
    if not isinstance(value, tuple | list):
        return ()
    dependencies = tuple(value)
    assert all(isinstance(item, str) and item for item in dependencies), (
        "EVAL_DEPENDENCIES entries must be non-empty strings"
    )
    return dependencies


def _load_allowlist() -> dict:
    return tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def _undeclared_recipes_by_key(
    recipes: list[RegisteredAnalysisRecipe],
) -> dict[tuple[str, str, str], RegisteredAnalysisRecipe]:
    return {
        recipe.key: recipe
        for recipe in recipes
        if not recipe.declares_eval_dependency
    }


def _allowlist_by_key(allowlist: dict) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (
            entry["analysis_type"],
            entry["recipe_module"],
            entry["recipe_name"],
        ): entry
        for entry in allowlist.get("eval_dependency_exemptions", [])
    }


def _assert_no_unlisted_undeclared_recipes(
    found: dict[tuple[str, str, str], RegisteredAnalysisRecipe],
    allowed: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    new_recipes = sorted(set(found) - set(allowed))
    assert not new_recipes, (
        "Registered rlrmp analysis recipe(s) lack declared evaluation dependencies "
        f"and are not exempted in {ALLOWLIST_PATH.relative_to(REPO_ROOT)}: "
        f"{new_recipes}. Declare EVAL_DEPENDENCIES on the recipe function/module, "
        "or add a deliberate exemption with owner and reason."
    )
