"""Matrix-experiment analysis recipes."""

from rlrmp.analysis.matrix.history_payload import (
    HISTORY_PAYLOAD_ANALYSIS_TYPE,
    register_history_payload_recipe,
)
from rlrmp.analysis.matrix.standard_matrix import (
    STANDARD_MATRIX_ANALYSIS_TYPE,
    STANDARD_MATRIX_EVALUATION_TYPE,
    register_standard_matrix_recipes as _register_standard_matrix_recipes,
)


def register_standard_matrix_recipes(*, replace: bool = True) -> None:
    """Register matrix payload recipes, including generic history payloads."""
    _register_standard_matrix_recipes(replace=replace)
    register_history_payload_recipe(replace=replace)


__all__ = [
    "HISTORY_PAYLOAD_ANALYSIS_TYPE",
    "STANDARD_MATRIX_ANALYSIS_TYPE",
    "STANDARD_MATRIX_EVALUATION_TYPE",
    "register_standard_matrix_recipes",
]
