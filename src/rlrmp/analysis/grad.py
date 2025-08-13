"""
Compute derivatives of functions. 
"""

from collections.abc import Callable, Sequence
import re
from typing import Any, Optional

import equinox as eqx
import jax
import jax.tree as jt

import jax_cookbook.tree as jtree

from rlrmp.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, DefaultFigParamNamespace, FigParamNamespace, InputOf
from rlrmp.types import AnalysisInputData


_Tuple = jtree.make_named_tuple_subclass("Tuple")


class CallerPorts(AbstractAnalysisPorts):
    """Input ports for analyses which call other functions with positional arguments."""
    funcs: InputOf[Callable]
    func_args: tuple[InputOf[Any], ...]


def _compute_grads(
    grad_func: Callable, 
    funcs: Sequence[Callable], 
    func_args: tuple, 
    argnums: Optional[int | Sequence[int]],
) -> tuple:
    if argnums is None:
        argnums = tuple(range(len(func_args)))
    elif isinstance(argnums, int):
        argnums = (argnums,)

    grads_raw = jt.map(
        lambda func, *args: _Tuple(grad_func(func, *args, argnums=argnums)),
        funcs,
        *func_args,
    )

    grads_by_argnum = jtree.unzip(grads_raw, tuple_cls=_Tuple)

    grads_expanded: list = [
        grads_by_argnum[argnums.index(i)] if i in argnums else None
        for i in range(len(func_args))
    ]

    return type(func_args)(*grads_expanded)


#! TODO: `Jacobians` and `Hessians` seem like good candidates for 
#! refactoring by a simpler `AbstractAnalysis` functional constructor
class Jacobians(AbstractAnalysis[CallerPorts]):
    Ports = CallerPorts
    inputs: CallerPorts = eqx.field(default_factory=CallerPorts, converter=CallerPorts.converter) 
    
    variant: Optional[str] = "full"

    argnums: Optional[int | Sequence[int]] = None

    def compute(
        self,
        data: AnalysisInputData,
        *,
        funcs,
        func_args: tuple,
        **kwargs,
    ) -> tuple:
        return _compute_grads(
            lambda func, *args, argnums: jax.jacobian(func, argnums=argnums)(*args),
            funcs, 
            func_args,
            self.argnums,
        )


class Hessians(AbstractAnalysis[CallerPorts]):
    Ports = CallerPorts
    inputs: CallerPorts = eqx.field(default_factory=CallerPorts, converter=CallerPorts.converter)

    variant: Optional[str] = "full"

    argnums: Optional[Sequence[int]] = None
    diag_only: bool = True  # Whether to compute only the diagonal Hessians (i.e. no cross-input terms)

    def compute(
        self,
        data: AnalysisInputData,
        *,
        funcs,
        func_args,
        **kwargs,
    ):
        def get_hessians(func, *args, argnums):
            if self.diag_only:
                return [jax.hessian(func, argnums=i)(*args) for i in argnums]
            else: 
                return jax.hessian(func, argnums=argnums)(*args)

        return _compute_grads(get_hessians, funcs, func_args, self.argnums)