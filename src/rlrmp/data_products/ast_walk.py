"""Shared AST traversal primitives for numeric data-in-code gates."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator


def walk_numeric_nodes(
    node: ast.AST,
    *,
    is_numeric: Callable[[ast.AST], bool],
) -> Iterator[ast.AST]:
    """Yield numeric nodes found by a complete AST walk."""

    return (child for child in ast.walk(node) if is_numeric(child))
