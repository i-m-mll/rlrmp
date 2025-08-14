from rlrmp.analysis.analysis import AbstractAnalysis, SinglePort
from rlrmp.types import AnalysisInputData


import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree


from collections.abc import Callable
from typing import Any, Optional


class ApplyFuncs(AbstractAnalysis[SinglePort[PyTree]]):
    """Apply a PyTree of callables to a PyTree of data.

    The functions are stored in the instance field `funcs` and are applied to
    the leaves of the `data` PyTree (customizable via `is_data_leaf`).

    - If `funcs` is not structurally aligned with `data`, set
      `expand_funcs_to_data=True` to prefix-expand functions across the
      structure of `data` (useful when one set of functions is broadcast over
      many conditions).
    - By default, callable/equinox-`Measure` leaves are treated as function
      leaves; `Responses` or arrays are treated as data leaves.
    """

    Ports = SinglePort
    inputs: SinglePort[PyTree] = eqx.field(
        default_factory=SinglePort[PyTree], converter=SinglePort[PyTree].converter
    )

    funcs: PyTree[Callable] = eqx.field(kw_only=True)  # required at runtime
    is_leaf: Optional[Callable[[Any], bool]] = None

    def _apply_func(self, func, subdata):
        return jt.map(lambda leaf: func(leaf), subdata, is_leaf=self.is_leaf)

    def compute(self, data: AnalysisInputData, *, input: PyTree, **kwargs):
        result = jt.map(
            lambda func: jt.map(
                lambda x: func(x),
                input,
                is_leaf=self.is_leaf,
            ),
            self.funcs,
            is_leaf=callable,
        )
        return result