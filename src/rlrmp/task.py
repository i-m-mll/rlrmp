from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from equinox import field
from feedbax import (
    DelayedReaches,
    DelayedReachTaskInputs,
    SimpleReaches,
    TaskTrialSpec,
    TrialTimeline,
    WhereDict,
    centreout_endpoints,
    forceless_task_inputs,
    gen_epoch_lengths,
    get_masked_seqs,
    get_masks,
    get_scalar_epoch_seq,
    pos_only_states,
)
from feedbax.loss import TargetSpec
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
        effector_init_state, effector_target_state = pos_only_states(self._fixed_endpoints())
        effector_target_state = jt.map(
            lambda x: jnp.broadcast_to(x, (self.n_steps - 1, *x.shape)),
            effector_target_state,
        )
        return self._construct_trial_spec(effector_init_state, effector_target_state)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        """Return one validation trial with the same fixed endpoint pair."""

        del key
        endpoints = jnp.stack([self.fixed_init_pos[None, :], self.fixed_target_pos[None, :]])
        effector_init_states, effector_target_states = pos_only_states(endpoints)
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
        effector_init_state, effector_target_state = pos_only_states(endpoints)

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

        effector_init_states, effector_target_states = pos_only_states(endpoints)

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


class CsDelayedCenterOutReaches(CenterOutDelayedReaches):
    """C&S delayed reach with target visible throughout prep and movement.

    This task interprets ``n_steps`` as the number of rollout/control stages,
    matching the C&S LSS graph runtime. The preparation epoch length is sampled
    from ``epoch_len_ranges``; the remaining stages are movement.
    """

    epoch_names: Tuple[str, ...] = ("prep", "movement")
    epoch_len_ranges: Tuple[Tuple[int, int], ...] = ((10, 31),)
    hold_epochs: Tuple[int, ...] = (0,)
    target_on_epochs: Tuple[int, ...] = (0, 1)
    move_epochs: Tuple[int, ...] = (1,)
    p_catch_trial: float = 0.0
    train_endpoint_mode: str = "center_out"

    def _get_sequences(
        self,
        init_states,
        target_states,
        key: PRNGKeyArray,
        *,
        p_catch: float,
    ):
        del p_catch
        n_time = int(self.n_steps)
        epoch_lengths_pre = gen_epoch_lengths(key, self.epoch_len_ranges)
        remaining_len = n_time - jnp.sum(epoch_lengths_pre)
        remaining_len = jnp.maximum(remaining_len, 0)
        epoch_lengths = jnp.concatenate((epoch_lengths_pre, jnp.array([remaining_len])))
        epoch_bounds = jnp.pad(jnp.cumsum(epoch_lengths), (1, 0), constant_values=(0, -1))
        epoch_masks = get_masks(n_time, epoch_bounds)
        move_epochs = jnp.asarray(self.move_epochs, dtype=jnp.int32)
        hold_epochs = jnp.asarray(self.hold_epochs, dtype=jnp.int32)
        target_on_epochs = jnp.asarray(self.target_on_epochs, dtype=jnp.int32)

        target_seqs = jt.map(
            lambda target, init: target + init,
            get_masked_seqs(target_states, epoch_masks[move_epochs]),
            get_masked_seqs(init_states, epoch_masks[hold_epochs]),
        )
        visible_target = jt.map(
            lambda x: jnp.broadcast_to(x, (n_time, *x.shape)),
            forceless_task_inputs(target_states),
        )
        stim_on_seq = get_scalar_epoch_seq(epoch_bounds, n_time, 1.0, target_on_epochs)
        hold_seq = get_scalar_epoch_seq(epoch_bounds, n_time, 1.0, hold_epochs)
        return DelayedReachTaskInputs(visible_target, hold_seq, stim_on_seq), target_seqs, epoch_bounds

    def get_train_trial(
        self, key: PRNGKeyArray, batch_info=None
    ) -> TaskTrialSpec:
        """Center-out delayed reach from origin in a uniformly random direction."""

        del batch_info
        key_dir, key_seq = jr.split(key)
        endpoints = _random_centerout_endpoints(key_dir, self.eval_reach_length)
        effector_init_state, effector_target_state = pos_only_states(endpoints)
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
        """Center-out delayed validation reaches from origin."""

        del key
        origin = jnp.zeros(2)
        endpoints = centreout_endpoints(
            origin, self.eval_n_directions, self.eval_reach_length
        )
        effector_init_states, effector_target_states = pos_only_states(endpoints)
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


TASK_TYPES = {
    "simple_reach": EpochSimpleReaches,
    "fixed_simple_reach": FixedEpochSimpleReaches,
    "delayed_reach": DelayedReaches,
    "center_out_delayed_reach": CenterOutDelayedReaches,
    "cs_delayed_center_out_reach": CsDelayedCenterOutReaches,
}
