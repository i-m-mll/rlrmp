
from collections.abc import Callable, Mapping
from copy import deepcopy
import inspect
from inspect import Parameter
import logging
from operator import is_
from re import A
from types import MappingProxyType
from typing import Any, Concatenate, Generic, Optional, ParamSpec, Self, TypeVar

import equinox as eqx
import jax.tree as jt
from jaxtyping import Array, PyTree
import plotly.graph_objects as go

import feedbax.plotly as fbp

from rlrmp.config.config import PLOTLY_CONFIG
from rlrmp.hyperparams import flat_key_to_where_func
from rlrmp.misc import deep_merge
from rlrmp.tree_utils import ldict_level_to_bottom
from rlrmp.types import AnalysisInputData, LDict
from rlrmp.plot_utils import get_label_str
from rlrmp.analysis.analysis import (
    AbstractAnalysis,
    PortsType,
    SinglePort,
)


logger = logging.getLogger(__name__)


def _validate_defaults_against_callable(
    cls: type, fn: Callable[..., Any], 
    defaults: Mapping[str, Any]
) -> None:
    """Raise at import time if subclass `fig_params` has keys not accepted by `fig_fn`."""
    sig = inspect.signature(fn)
    params = sig.parameters

    # If the function accepts **kwargs, allow any keys
    has_var_kw = any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return

    # Otherwise only these names are legal as keyword arguments:
    legal = {
        name for name, p in params.items()
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }

    # Enforce identifier-ness (so they can be shown as parameters)
    bad_ident = [k for k in defaults if not str(k).isidentifier()]
    if bad_ident:
        raise TypeError(f"{cls.__name__}.fig_params contains non-identifier keys: {bad_ident!r}")

    extraneous = [k for k in defaults if k not in legal]
    if extraneous:
        raise TypeError(
            f"{cls.__name__}.fig_params contains keys not accepted by {getattr(fn, '__name__', fn)}: "
            f"{extraneous!r}. Legal kwargs: {sorted(legal)}"
        )


FigFnKwargs = ParamSpec('FigFnKwargs')


class AbstractPlotter(AbstractAnalysis[PortsType], Generic[PortsType, FigFnKwargs]):
    """Base class for analyses that produce plotly figures.
    
    Note:
        By making this class and its subclasses generics of `FigFnKwargs`, type inference is 
        supplied for the signature of `with_fig_params`. When subclassing, it is necessary to 
        properly bind `FigFnKwargs` to the signature of the `fig_fn` field. See `ScatterN2D` for 
        an example. 
    """
    
    fig_fn: Callable[Concatenate[..., FigFnKwargs], go.Figure] = eqx.field(kw_only=True) 

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        # Try to obtain a callable to validate against. Subclasses usually set a default.
        fn = getattr(cls, "fig_fn", None)
        if isinstance(fn, staticmethod):           # just in case someone uses staticmethod
            fn = fn.__func__
        if not callable(fn):
            # Can't validate or synthesize a signature without a callable.
            # We'll do a runtime check in __post_init__ instead (below).
            return

        _validate_defaults_against_callable(cls, fn, cls.fig_params)

    def __post_init__(self):
        """If a subclass didn't provide a class-level `fig_fn`, validate at instance time."""
        fn = self.fig_fn
        if callable(fn):
            _validate_defaults_against_callable(type(self), fn, type(self).fig_params)

    def with_fig_params(self, *args: FigFnKwargs.args, **kwargs: FigFnKwargs.kwargs) -> Self:
        """Returns a copy of this analysis with updated figure parameters."""
        if any(args):
            logger.warning(
                "No positional args should be given to `with_fig_params`; if your type checker / "
                "IDE suggested you should provide a positional arg, the cause is likely that the "
                f"signature of `fig_fn` was not properly bound in {type(self).__name__}."
            )
        return eqx.tree_at(
            lambda x: x.fig_params,
            self,
            MappingProxyType(deep_merge(self.fig_params, kwargs)),
        )
        

class ScatterN2D(AbstractPlotter[SinglePort, FigFnKwargs]):
    """General 2D plotter (lines/markers) over a PyTree of arrays.
    
    Example:
      Use the output of `AlignedVars` as input. Set `subplot_level` to "vars" to get one 
      subplot per aligned variable. Set `colorscale_axis=0` and `colorscale_key="pert__amp"`,
      and modify the instance with `after_stacking("pert__amp")` to get one series/curve per 
      "pert__amp" value on each subplot. 
    
    Fields:
      subplot_level: If set, maps keys at this level to subplots. 
      colorscale_axis: Which axis of the array leaves to plot as separate series. 
      colorscale_key: If set, determines the colorscale to use across the series.
    """
    Ports = SinglePort[Array]
    inputs: SinglePort[Array] = eqx.field(  # pyright: ignore[reportGeneralTypeIssues]
        kw_only=True, converter=SinglePort[Array].converter
    )
    fig_fn: Callable[Concatenate[PyTree, FigFnKwargs], go.Figure] \
        = fbp.trajectories_2D  # pyright: ignore[reportAssignmentType]  

    subplot_level: str | None = None
    # colorscale_key: Optional[str] = None

    # Node-level default fig parameters; everything is optional.
    fig_params: Mapping[str, Any] = MappingProxyType(dict(
        colorscale_key=None,
        colorscale_axis=0,
        # colorscale=PLOTLY_CONFIG.default_colorscale,
        curves_mode="lines",                  # "lines" | "markers" | "lines+markers"
        darken_mean=PLOTLY_CONFIG.mean_lighten_factor,
        n_curves_max=20,
        layout_kws=dict(
            width=900,
            height=300,
            legend_tracegroupgap=1,
            margin_t=50,
            margin_b=20,
        ),
        scatter_kws=dict(
            line_width=0.5,
            opacity=0.5,
        ),
        mean_scatter_kws=dict(
            line_width=2.5,
            opacity=1,
        ),
    ))

    def make_figs(
        self, 
        data: AnalysisInputData, 
        *, 
        input, 
        colorscales, 
        hps_common, 
        **kwargs,
    ) -> PyTree[go.Figure]:
        fig_params = self.fig_params

        colorscale_key = fig_params.get("colorscale_key")
        if colorscale_key is not None:
            updates = {}
            if fig_params.get("colorscale") is None:
                updates["colorscale"] = colorscales[colorscale_key]
            if fig_params.get("legend_title") is None:
                updates["legend_title"] = get_label_str(colorscale_key)
            if fig_params.get("legend_labels") is None:
                updates["legend_labels"] = flat_key_to_where_func(colorscale_key)(hps_common)
            
            if updates:
                fig_params = deep_merge(fig_params, updates)

        def _make_fig(node_data: PyTree[Array]) -> go.Figure:
            return fbp.trajectories_2D(node_data, **fig_params)

        # one_series can be a single array OR an LDict of leaves (arrays -> subplots) 
        if self.subplot_level is not None:
            input = ldict_level_to_bottom(self.subplot_level, input)
            is_leaf = LDict.is_of(self.subplot_level)
        else:
            is_leaf=None
        
        return jt.map(_make_fig, input, is_leaf=is_leaf)
    

class ScatterN3D(AbstractPlotter[SinglePort, FigFnKwargs]):
    """General 3D plotter (lines/markers) over a PyTree of 3D arrays."""
    Ports = SinglePort[Array]
    inputs: SinglePort[Array] = eqx.field(  # pyright: ignore[reportGeneralTypeIssues]
        kw_only=True, converter=Ports.converter
    )
    fig_fn: Callable[Concatenate[PyTree, FigFnKwargs], go.Figure] \
        = fbp.trajectories_3D  # pyright: ignore[reportAssignmentType]  
        
    colorscale_key: Optional[str] = None
    colorscale_axis: int = 0

    fig_params: Mapping = MappingProxyType(dict(
        axis_labels=("PC1", "PC2", "PC3"),
        mode="lines",                # "lines" | "markers" (fbp.trajectories_3D)
        name=None,                   # optional legend name for this series
        # marker_size=..., line_width=..., endpoint_symbol=..., etc.
    ))

    def make_figs(
        self, data: AnalysisInputData, *, input, colorscales=None, **kwargs
    ) -> PyTree[go.Figure]:
        fig_params = self.fig_params

        colorscale = None
        if self.colorscale_key is not None and colorscales is not None:
            colorscale = colorscales[self.colorscale_key]
            if fig_params.get("legend_title") is None:
                fig_params = MappingProxyType(deep_merge(fig_params, {"legend_title": get_label_str(self.colorscale_key)}))

        def _make_fig(arr3d: Array | PyTree[Array]) -> go.Figure:
            return fbp.trajectories_3D(
                arr3d,
                colorscale=colorscale,
                colorscale_axis=self.colorscale_axis,
                **fig_params,
            )

        # Typically a single array; if a tree, the calling code will map before we get here.
        return jt.map(_make_fig, input)