from feedbax import (
    DelayedReaches,
    SimpleReaches,
)


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


TASK_TYPES = {
    "simple_reach": SimpleReaches,
    "fixed_simple_reach": SimpleReaches,
    "delayed_reach": delayed_reaches,
    "center_out_delayed_reach": center_out_delayed_reaches,
    "cs_delayed_center_out_reach": cs_delayed_center_out_reaches,
}
