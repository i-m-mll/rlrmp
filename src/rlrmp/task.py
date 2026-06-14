import jax.numpy as jnp
import jax.tree as jt
from equinox import field
from feedbax import (
    DelayedReaches,
    SimpleReaches,
    TaskTrialSpec,
    TrialTimeline,
    pos_only_states,
)
from feedbax.objectives.loss import TargetSpec
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

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
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


def _delayed_center_out_reaches(*, loss_func, **kwargs) -> DelayedReaches:
    """Build the Feedbax delayed-center-out preset from RLRMP task specs."""

    kwargs = dict(kwargs)
    n_control_stages = kwargs.pop("n_control_stages", None)
    n_steps = kwargs.pop("n_steps", None)
    if n_control_stages is None:
        if n_steps is None:
            raise ValueError(
                "Delayed center-out tasks require either n_control_stages or n_steps."
            )
        n_control_stages = int(n_steps) - 1
    return DelayedReaches.delayed_center_out(
        loss_func=loss_func,
        n_control_stages=int(n_control_stages),
        **kwargs,
    )


def delayed_reaches(*, loss_func, **kwargs) -> DelayedReaches:
    """Build Feedbax ``DelayedReaches``, applying public presets when requested."""

    if kwargs.get("preset") == "delayed_center_out":
        return _delayed_center_out_reaches(loss_func=loss_func, **kwargs)
    kwargs = dict(kwargs)
    kwargs.pop("n_control_stages", None)
    return DelayedReaches(loss_func=loss_func, **kwargs)


def center_out_delayed_reaches(*, loss_func, **kwargs) -> DelayedReaches:
    """Compatibility constructor for historical ``center_out_delayed_reach`` specs."""

    kwargs = dict(kwargs)
    kwargs.setdefault("train_endpoint_mode", "center_out")
    return delayed_reaches(loss_func=loss_func, **kwargs)


def cs_delayed_center_out_reaches(*, loss_func, **kwargs) -> DelayedReaches:
    """Compatibility constructor for historical C&S delayed center-out specs."""

    kwargs = dict(kwargs)
    kwargs.setdefault("preset", "delayed_center_out")
    kwargs.setdefault("train_endpoint_mode", "center_out")
    kwargs.setdefault("epoch_names", ("prep", "movement"))
    kwargs.setdefault("target_on_epochs", (0, 1))
    kwargs.setdefault("hold_epochs", (0,))
    kwargs.setdefault("move_epochs", (1,))
    kwargs.setdefault("target_visible_from_start", True)
    kwargs.setdefault("go_cue_event_name", "go_cue")
    kwargs.setdefault("catch_metadata_policy", "flag")
    return _delayed_center_out_reaches(loss_func=loss_func, **kwargs)


CenterOutDelayedReaches = center_out_delayed_reaches
CsDelayedCenterOutReaches = cs_delayed_center_out_reaches


TASK_TYPES = {
    "simple_reach": EpochSimpleReaches,
    "fixed_simple_reach": FixedEpochSimpleReaches,
    "delayed_reach": delayed_reaches,
    "center_out_delayed_reach": center_out_delayed_reaches,
    "cs_delayed_center_out_reach": cs_delayed_center_out_reaches,
}
