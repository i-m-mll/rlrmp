"""Schema-generated command-line parsing for registered training configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel


def _literal_values(annotation: Any) -> list[Any]:
    origin = get_origin(annotation)
    if origin is Literal:
        return [value for value in get_args(annotation) if value is not None]
    if origin in {Union, UnionType}:
        values: list[Any] = []
        for member in get_args(annotation):
            values.extend(_literal_values(member))
        return values
    return []


def _argument_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        members = [member for member in get_args(annotation) if member is not type(None)]
        if len(members) == 1:
            return _argument_type(members[0])
    values = _literal_values(annotation)
    if values:
        return type(values[0])
    if annotation in {str, int, float, Path}:
        return annotation
    return str


def build_config_parser(
    model: type[BaseModel],
    *,
    description: str,
) -> argparse.ArgumentParser:
    """Generate one parser directly from a Pydantic training-config model."""

    parser = argparse.ArgumentParser(description=description)
    defaults = model().model_dump(mode="python")
    for name, field in model.model_fields.items():
        flag = f"--{name.replace('_', '-')}"
        default = defaults[name]
        options: dict[str, Any] = {
            "dest": name,
            "default": default,
            "help": f"Canonical config field {name} (default: {default!r}).",
        }
        if field.annotation is bool or isinstance(default, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = _argument_type(field.annotation)
            choices = _literal_values(field.annotation)
            if choices:
                options["choices"] = choices
        parser.add_argument(flag, **options)
    return parser


def parse_config(
    model: type[BaseModel],
    argv: list[str] | None = None,
    *,
    description: str,
) -> BaseModel:
    """Parse command-line arguments and return the validated typed config."""

    namespace = build_config_parser(model, description=description).parse_args(argv)
    return model.model_validate(vars(namespace))


__all__ = ["build_config_parser", "parse_config"]
