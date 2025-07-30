from collections.abc import Callable
from rlrmp.analysis.analysis import AbstractAnalysis, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace, NoPorts
from rlrmp.types import TreeNamespace

import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PyTree
from sklearn.decomposition import PCA

from feedbax.bodies import SimpleFeedbackState
from feedbax.misc import batch_reshape
from jax_cookbook import is_type
import jax_cookbook.tree as jtree

from typing import Optional


class StatesPCA(AbstractAnalysis[NoPorts]):
    """Perform principle component analysis on evaluated states, with sklearn.
    
    Assumes that `data.states` is an array PyTree of shape `(*batch, n_vars)`.
    After flattening the batch axes, PCA is fit on the concatenated leaves of the PyTree;
    thus the user should ensure that `n_vars` is the same for all leaves, either by 
    passing an appropriate value for the `where_states` field, or by transforming the 
    instance of this class with an appropriate prep op.

    Fields:
        n_components: Number of principal components to keep.
        where_states: Optional[Callable[[SimpleFeedbackState], PyTree[Array]]] = None

    Returns: A `TreeNamespace` with the following attributes:
        pca: The fitted sklearn PCA object.
        batch_transform: A function that applies the PCA transform to a PyTree of arrays of shape
            `(*batch, n_vars)`, retaining the batch axes in the output. 
    """
    
    n_components: Optional[int] = None
    where_states: Optional[Callable[[SimpleFeedbackState], PyTree[Array]]] = None
    return_data: bool = True

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        if self.where_states is not None: 
            states_for_pca = jt.map(self.where_states, data.states, is_leaf=is_type(SimpleFeedbackState))
        else:
            states_for_pca = data.states

        states_reshaped = jt.map(lambda arr: arr.reshape(-1, arr.shape[-1]), states_for_pca)

        #! TODO: Use JAX-native PCA
        pca = PCA(n_components=self.n_components).fit(
            jnp.concatenate(jt.leaves(states_reshaped))  # type: ignore
        )
        
        batch_transform = lambda x: jt.map(batch_reshape(pca.transform), x)  # type: ignore

        if self.return_data:
            states_pc = jt.map(batch_transform, states_for_pca)
        else:
            states_pc = None

        return TreeNamespace(
            pca=pca,
            batch_transform=batch_transform,
            states_pc=states_pc,
        )
