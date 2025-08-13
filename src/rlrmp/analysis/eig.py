from functools import partial
from typing import Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
import jax.tree as jt
import plotly.graph_objects as go
from jax_cookbook import is_type
from jaxtyping import PyTree, Float, Array

import jax_cookbook.tree as jtree

from rlrmp.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf
from rlrmp.types import AnalysisInputData, TreeNamespace


class DecompPorts(AbstractAnalysisPorts):
    matrices: InputOf[Float[Array, "... m n"]]
    

class SquareDecompPorts(AbstractAnalysisPorts):
    """Input ports for Eigendecomposition analysis."""
    matrices: InputOf[Float[Array, "... m m"]]


class Eig(AbstractAnalysis[SquareDecompPorts]):
    Ports = SquareDecompPorts
    inputs: SquareDecompPorts = field(default_factory=SquareDecompPorts, converter=SquareDecompPorts.converter)

    # @partial(jax.jit, device=jax.devices('cpu')[0])
    def _eig_cpu(self, *a, **kw):
        return tuple(jax.lax.linalg.eig(*a, **kw))

    def compute(
        self,
        data: AnalysisInputData,
        *,
        matrices,
        **kwargs,
    ):
        eigvals, eigvecs_l, eigvecs_r = jtree.unzip(jt.map(self._eig_cpu, matrices))
        return TreeNamespace(
            eigvals=eigvals,
            eigvecs_l=eigvecs_l,
            eigvecs_r=eigvecs_r,
        )



class SVD(AbstractAnalysis[DecompPorts]):
    Ports = DecompPorts 
    inputs: DecompPorts = field(default_factory=DecompPorts, converter=DecompPorts.converter)
    
    def compute(
        self,
        data: AnalysisInputData,
        *,
        matrices,
        **kwargs,
    ):
        singvecs_l, singvals, singvecs_r_adj = jtree.unzip(jt.map(jax.lax.linalg.svd, matrices))
        return TreeNamespace(
            singvals=singvals,
            singvecs_l=singvecs_l,
            singvecs_r_adj=singvecs_r_adj,
        )


def complex_to_polar_abs_angle(arr: Array) -> Array:
  """
  Converts complex numbers to polar coordinates with symmetric angles.
  
  This is useful when working with the distribution of eigenvalues of discrete 
  systems, where the absolute angle describes frequency and the magnitude describes 
  stability. 

  Args:
    arr: A JAX array of complex numbers with shape (..., n).

  Returns:
    A tuple containing two JAX arrays:
    - angles: The angles in radians, mapped to the interval [0, pi],
              symmetric about the real axis. Shape (..., n).
    - magnitudes: The magnitudes (radii) of the complex numbers.
                  Shape (..., n).
  """
  # Calculate magnitudes (absolute values)
  magnitudes = jnp.abs(arr)
  # Calculate standard angles in (-pi, pi] and take absolute value for [0, pi]
  angles = jnp.abs(jnp.angle(arr))
  return jnp.stack([angles, magnitudes], axis=0)