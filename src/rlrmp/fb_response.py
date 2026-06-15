from collections.abc import Callable

import equinox as eqx
import jax.tree as jt
import jax_cookbook.tree as jtree
from equinox import field
from feedbax.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    Data,
    InputOf,
)
from rlrmp.misc import unit_circle_points
from jax_cookbook import LDict
from jax_cookbook import is_module
from jaxtyping import ArrayLike, PyTree


class InstantFBResponsePorts(AbstractAnalysisPorts):
    # Note: In eager-models architecture, path is m.net.hidden (no .step indirection)
    rnn_cells: InputOf[Callable] = Data.models(where=lambda m: m.net.hidden)


class InstantFBResponse(AbstractAnalysis[InstantFBResponsePorts]):
    """Plot the instantaneous feedback response of the network."""

    Ports = InstantFBResponsePorts
    inputs: InstantFBResponsePorts = eqx.field(default_factory=Ports, converter=Ports.converter)

    n_directions: int = 24
    # n_samples: int = 1000
    fb_pert_amp: PyTree[float] = 0.5
    input_idxs: PyTree[slice | ArrayLike] = field(
        default_factory=lambda: LDict.of("fb_var")(
            pos=slice(5, 7),
            vel=slice(7, 9),
        )
    )

    def compute(self, data, *, rnn_cells, hps_common, **kwargs):
        fb_perts = self.fb_pert_amp * unit_circle_points(self.n_directions)
        pert_step = hps_common.pert.unit.start_step

        def _compute_single(rnn_cell, states):
            base_net_state = jtree.first_leaf(states, is_leaf=is_module).net
            # TODO: Repeat computation independently over multiple steady-state steps, and average.
            init_state = base_net_state.hidden[..., pert_step - 1, :]
            base_input = base_net_state.input[..., pert_step, :]

            def _single_pert(fb_pert):
                def _per_input(idxs):
                    perturbed_input = base_input.at[..., idxs].add(fb_pert)
                    return rnn_cell(perturbed_input, init_state) - init_state

                return jt.map(_per_input, self.input_idxs)

            return eqx.filter_vmap(_single_pert)(fb_perts)

        return jt.map(
            _compute_single,
            rnn_cells,
            data.states,
            is_leaf=is_module,
        )
