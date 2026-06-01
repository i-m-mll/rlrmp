from functools import partial
from typing import Optional, Tuple

from equinox import field
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
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


class EpochSimpleReaches(SimpleReaches):
    """Simple reaches with a full-trial movement epoch for RLRMP losses."""

    def _construct_trial_spec(self, effector_init_state, effector_target_state):
        trial_spec = super()._construct_trial_spec(effector_init_state, effector_target_state)
        n_time = self.n_steps - 1
        epoch_bounds = jnp.asarray((0, n_time), dtype=jnp.int32)
        targets = trial_spec.targets
        if effector_init_state.pos.ndim > 1:
            batch_size = effector_init_state.pos.shape[0]
            epoch_bounds = jnp.broadcast_to(epoch_bounds, (effector_init_state.pos.shape[0], 2))
            targets = jt.map(
                lambda spec: (
                    TargetSpec(
                        value=spec.value,
                        time_idxs=spec.time_idxs,
                        time_mask=spec.time_mask,
                        discount=(
                            jnp.broadcast_to(spec.discount, (batch_size, n_time))
                            if spec.discount is not None
                            and hasattr(spec.discount, "shape")
                            and spec.discount.shape == (n_time,)
                            else spec.discount
                        ),
                    )
                    if isinstance(spec, TargetSpec)
                    else spec
                ),
                targets,
                is_leaf=lambda x: isinstance(x, TargetSpec),
            )
        return TaskTrialSpec(
            inits=trial_spec.inits,
            inputs=trial_spec.inputs,
            targets=targets,
            intervene=trial_spec.intervene,
            extra=trial_spec.extra,
            timeline=TrialTimeline.from_epochs_events(
                n_time,
                epoch_bounds=epoch_bounds,
                epoch_names=("movement",),
            ),
        )


class FixedEpochSimpleReaches(EpochSimpleReaches):
    """Simple reach task with a fixed train/validation endpoint pair."""

    fixed_init_pos: Float[Array, "2"] = field(
        default_factory=lambda: jnp.zeros(2),
        converter=jnp.asarray,
    )
    fixed_target_pos: Float[Array, "2"] = field(
        default_factory=lambda: jnp.asarray([0.15, 0.0]),
        converter=jnp.asarray,
    )

    def _fixed_endpoints(self) -> Float[Array, "2 2"]:
        return jnp.stack([self.fixed_init_pos, self.fixed_target_pos])

    def get_train_trial(
        self, key: PRNGKeyArray, batch_info=None
    ) -> TaskTrialSpec:
        """Return the fixed endpoint pair, ignoring randomness."""

        del key, batch_info
        effector_init_state, effector_target_state = _pos_only_states(self._fixed_endpoints())
        effector_target_state = jt.map(
            lambda x: jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)),
            effector_target_state,
        )
        return self._construct_trial_spec(effector_init_state, effector_target_state)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        """Return one validation trial with the same fixed endpoint pair."""

        del key
        endpoints = jnp.stack([self.fixed_init_pos[None, :], self.fixed_target_pos[None, :]])
        effector_init_states, effector_target_states = _pos_only_states(endpoints)
        effector_target_states = jt.map(
            lambda x: jnp.swapaxes(
                jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)),
                0,
                1,
            ),
            effector_target_states,
        )
        return self._construct_trial_spec(effector_init_states, effector_target_states)

    @property
    def n_validation_trials(self) -> int:
        """Number of validation trials."""

        return 1


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
    "simple_reach": EpochSimpleReaches,
    "fixed_simple_reach": FixedEpochSimpleReaches,
    "delayed_reach": DelayedReaches,
    "center_out_delayed_reach": CenterOutDelayedReaches,
}
