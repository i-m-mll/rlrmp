from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.loss import TargetSpec
from feedbax._mapping import WhereDict
from feedbax.task import (
    DelayedReaches,
    SimpleReaches,
    TaskTrialSpec,
    TrialTimeline,
    _pos_only_states,
    centreout_endpoints,
)
from jaxtyping import Array, Float, PRNGKeyArray


def _random_centerout_endpoints(
    key: PRNGKeyArray,
    reach_length: float,
) -> Float[Array, "2 2"]:
    """Sample a single center-out reach from origin in a random direction.

    Returns:
        Array of shape (2, 2) where [0] is the start (origin) and [1] is the
        target at ``reach_length`` distance in a uniformly random direction.
    """
    angle = jr.uniform(key, (), minval=0, maxval=2 * jnp.pi)
    start = jnp.zeros(2)
    target = reach_length * jnp.stack([jnp.cos(angle), jnp.sin(angle)])
    return jnp.stack([start, target])


class CenterOutDelayedReaches(DelayedReaches):
    """Delayed reaching task with center-out reaches from the origin.

    All training reaches start at (0, 0) and end at a random direction at
    constant distance (``eval_reach_length``). This enforces translation
    invariance: absolute position is equivalent to position relative to the
    start, since the start is always the origin.

    Validation trials are also center-out from the origin, at
    ``eval_n_directions`` evenly-spaced directions.
    """

    def get_train_trial(
        self, key: PRNGKeyArray, batch_info=None
    ) -> TaskTrialSpec:
        """Center-out reach from origin in a uniformly random direction.

        Arguments:
            key: Random key for sampling the reach direction and epoch lengths.
        """
        key_dir, key_seq = jr.split(key)

        endpoints = _random_centerout_endpoints(key_dir, self.eval_reach_length)
        effector_init_state, effector_target_state = _pos_only_states(endpoints)

        task_inputs, effector_target_states, epoch_bounds = self._get_sequences(
            effector_init_state,
            effector_target_state,
            key_seq,
            p_catch=self.p_catch_trial,
        )

        return TaskTrialSpec(
            inits=WhereDict(
                {(lambda state: state.mechanics.effector): effector_init_state},
            ),
            inputs=task_inputs,
            targets=WhereDict(
                {
                    (lambda state: state.mechanics.effector.pos): (
                        TargetSpec(effector_target_states.pos)
                    ),
                }
            ),
            timeline=TrialTimeline.from_epochs_events(
                self.n_steps,
                epoch_bounds=epoch_bounds,
                epoch_names=self.epoch_names,
            ),
        )

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        """Center-out reach set from origin at evenly-spaced directions.

        Uses ``eval_n_directions`` directions and ``eval_reach_length`` radius.
        The eval_grid_n field is ignored; all reaches start from origin.
        """
        origin = jnp.zeros(2)
        endpoints = centreout_endpoints(
            origin, self.eval_n_directions, self.eval_reach_length
        )  # shape (2, eval_n_directions, 2)

        effector_init_states, effector_target_states = _pos_only_states(endpoints)

        key_val = jr.PRNGKey(self.seed_validation)
        epochs_keys = jr.split(key_val, effector_init_states.pos.shape[0])
        get_sequences = partial(self._get_sequences, p_catch=0.0)
        task_inputs, effector_target_states, epoch_bounds = jax.vmap(get_sequences)(
            effector_init_states, effector_target_states, epochs_keys
        )

        return TaskTrialSpec(
            inits=WhereDict(
                {(lambda state: state.mechanics.effector): effector_init_states},
            ),
            inputs=task_inputs,
            targets=WhereDict(
                {
                    (lambda state: state.mechanics.effector.pos): (
                        TargetSpec(effector_target_states.pos)
                    ),
                }
            ),
            timeline=TrialTimeline.from_epochs_events(
                self.n_steps,
                epoch_bounds=epoch_bounds,
                epoch_names=self.epoch_names,
            ),
        )

    @property
    def n_validation_trials(self) -> int:
        """Number of validation trials (one per direction, always from origin)."""
        return self.eval_n_directions


TASK_TYPES = {
    "simple_reach": SimpleReaches,
    "delayed_reach": DelayedReaches,
    "center_out_delayed_reach": CenterOutDelayedReaches,
}
