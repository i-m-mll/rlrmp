"""
Compute derivatives of functions. 
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional

import equinox as eqx
import jax
import jax.tree as jt

import jax_cookbook.tree as jtree

from rlrmp.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace, InputOf


_Tuple = jtree.make_named_tuple_subclass("Tuple")


class CallerPorts(AbstractAnalysisPorts):
    """Input ports for analyses which call other functions with positional arguments."""
    funcs: InputOf[Callable]
    func_args: tuple[InputOf[Any], ...]


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
        if self.argnums is None:
            argnums = tuple(range(len(func_args)))
        elif isinstance(self.argnums, int):
            argnums = (self.argnums,)
        else:
            argnums = self.argnums

        jacs_raw = jt.map(
            lambda func, *args: _Tuple(jax.jacobian(func, argnums=argnums)(*args)),
            funcs,
            *func_args,
        )

        jacs_for_argnums = jtree.unzip(jacs_raw, tuple_cls=_Tuple)

        jacs_expanded: list = [
            jacs_for_argnums[argnums.index(i)] if i in argnums else None
            for i in range(len(func_args))
        ]

        return type(func_args)(*jacs_expanded)


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
        if self.argnums is None:
            argnums = tuple(range(len(func_args)))
        elif isinstance(self.argnums, int):
            argnums = (self.argnums,)
        else:
            argnums = self.argnums

        def get_hessians(func, *args):
            if self.diag_only:
                return _Tuple(jax.hessian(func, argnums=i)(*args) for i in argnums)
            else: 
                return _Tuple(jax.hessian(func, argnums=argnums)(*args))


        hess_raw = jt.map(get_hessians, funcs, *func_args)

        hess_for_argnums = jtree.unzip(hess_raw, tuple_cls=_Tuple)

        hess_expanded: list = [
            hess_for_argnums[argnums.index(i)] if i in argnums else None
            for i in range(len(func_args))
        ]

        return type(func_args)(*hess_expanded)