"""Registry for rlrmp recipe params models.

This is a narrow bridge until Feedbax registration accepts a native
``params_model=`` keyword.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel


ParamsModel = type[BaseModel]

_PARAMS_MODELS: dict[str, ParamsModel] = {}


def register_params_model(
    recipe_name: str,
    model_class: ParamsModel,
    *,
    replace: bool = True,
) -> None:
    """Register the params model for one rlrmp recipe name."""

    existing = _PARAMS_MODELS.get(recipe_name)
    if existing is not None and existing is not model_class and not replace:
        raise ValueError(f"params model already registered for {recipe_name!r}")
    _PARAMS_MODELS[recipe_name] = model_class


def params_model_for(recipe_name: str) -> ParamsModel:
    """Return the params model registered for ``recipe_name``."""

    try:
        return _PARAMS_MODELS[recipe_name]
    except KeyError as exc:
        raise KeyError(f"no params model registered for {recipe_name!r}") from exc


def registered_params_models() -> Mapping[str, ParamsModel]:
    """Return all registered params models keyed by recipe name."""

    return MappingProxyType(dict(_PARAMS_MODELS))


__all__ = [
    "ParamsModel",
    "params_model_for",
    "register_params_model",
    "registered_params_models",
]
