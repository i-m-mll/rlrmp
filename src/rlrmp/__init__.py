"""RNNs Learn Robust Motor Policies experiment package."""

import logging
import sys

logger = logging.getLogger(__name__)

# Lazy-registration guard. Recipe registration imports the full feedbax + rlrmp
# public API, which is NOT available while `feedbax.__init__` is still running
# (plugin discovery is triggered mid-init by `feedbax.config`, before
# `feedbax.AbstractTask` and friends are defined). `register_experiment_package`
# therefore registers recipes only when feedbax has finished initializing;
# during the mid-init discovery pass it defers, and the deferred registration is
# completed lazily on first import of any `rlrmp.analysis` module (which only
# happens after `feedbax` has finished initializing). See issue 462bb31 (and the
# feedbax-side lazy-discovery fix in ccfe63a).
#
# States: ``None`` (not started), ``"in_progress"`` (running, used to break the
# re-entrant import cycle through ``rlrmp.analysis.__init__``), ``True`` (done).
_RECIPES_REGISTERED: "bool | str | None" = None


def _feedbax_public_api_ready() -> bool:
    """Return whether ``feedbax``'s public API is fully initialized.

    rlrmp's recipe-registration modules import the feedbax public API
    (``feedbax.AbstractTask`` and deeper ``feedbax.analysis.*`` symbols). During
    plugin discovery — which feedbax triggers *mid* its own ``__init__`` — those
    symbols do not exist yet. ``AbstractTask`` is bound near the end of
    ``feedbax.__init__``, so its presence is a reliable readiness signal.
    """
    feedbax = sys.modules.get("feedbax")
    return feedbax is not None and hasattr(feedbax, "AbstractTask")


def _module_initializing(name: str) -> bool:
    """Return whether a module is present but still executing its body."""

    module = sys.modules.get(name)
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


def register_experiment_package(registry):
    """Register this experiment package with the feedbax plugin registry.

    Plugin discovery runs this DURING ``feedbax.__init__`` (it is triggered by
    ``feedbax.config`` before the feedbax public API is fully defined), so the
    package-metadata registration is kept lightweight: it only imports the bare
    ``rlrmp`` package. The heavier recipe registration — which imports the full
    feedbax/rlrmp API — is attempted only when feedbax has finished initializing
    (:func:`ensure_rlrmp_recipes_registered`); during the mid-init discovery
    pass it is deferred to the import-time hook in ``rlrmp.analysis.__init__``,
    which fires on first real use, well after ``feedbax`` is ready.

    Args:
        registry: ExperimentRegistry instance to register with
    """
    from feedbax.plugins.discovery import register_package_from_module_info

    register_package_from_module_info(
        registry=registry,
        package_name="rlrmp",
        package_module_name="rlrmp",
        parts=[],
        analysis_module_root="analysis",
        training_module_root="train",
        config_resource_root="config",
        figure_routing={
            "spec_dir_template": "results/{experiment}/figures/{topic}",
            "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
            "spec_format": "json",
            "render_format": "html",
            "create_symlink_in_spec_dir": True,
        },
    )

    # Register recipes now if feedbax is ready; otherwise defer. We must NOT
    # touch the heavy ``rlrmp.analysis`` machinery while feedbax is mid-init:
    # the imports would fail AND would cache ``rlrmp.analysis`` in a state where
    # its registration side effect was skipped, defeating the lazy trigger.
    #
    # ``force=True``: an explicit ``register_experiment_package`` call is a
    # request to (re)install rlrmp's recipes into feedbax's recipe registry,
    # even if a previous call already did so. Tests (and any caller that swaps
    # recipes in and out of feedbax's process-global recipe registry) rely on
    # this to restore rlrmp's recipes. The underlying registrations are
    # idempotent (``replace=True``), so forcing is safe.
    ensure_rlrmp_recipes_registered(defer_if_feedbax_initializing=True, force=True)


def ensure_rlrmp_recipes_registered(
    *, defer_if_feedbax_initializing: bool = False, force: bool = False
) -> None:
    """Idempotently register rlrmp's spec families and analysis recipes.

    The recipe-registration modules import the full feedbax public API
    (``feedbax.analysis.*``, ``feedbax.contracts.*``), so this can only run once
    ``feedbax`` has finished initializing. It is the deferred tail of plugin
    registration (see :func:`register_experiment_package`) and is also invoked as
    an import-time side effect of ``rlrmp.analysis.__init__``. By default it is a
    no-op after the first successful run.

    Args:
        defer_if_feedbax_initializing: When ``True``, skip registration (leaving
            it for a later call) if ``feedbax`` is still mid-initialization.
            Used by :func:`register_experiment_package`, which runs during
            ``feedbax.__init__``. Direct callers and the ``rlrmp.analysis``
            import hook leave this ``False`` so registration always proceeds.
        force: When ``True``, (re)register even if a previous call already
            succeeded. The underlying registrations use ``replace=True`` so this
            is safe; it lets an explicit :func:`register_experiment_package` call
            restore rlrmp's recipes into feedbax's process-global recipe
            registry after another caller has swapped them out. The re-entrancy
            guard (``"in_progress"``) is still respected.

    On failure the registered-flag is reset so a later call re-attempts
    registration.
    """
    global _RECIPES_REGISTERED
    if _RECIPES_REGISTERED == "in_progress":
        return
    if _RECIPES_REGISTERED is True and not force:
        return

    if defer_if_feedbax_initializing and not _feedbax_public_api_ready():
        logger.debug("Deferring rlrmp recipe registration; feedbax not fully initialized.")
        return
    initializing_train_modules = (
        "rlrmp.train.task_model",
        "rlrmp.train.cs_perturbation_training",
        "rlrmp.train.cs_nominal_gru",
    )
    if any(_module_initializing(name) for name in initializing_train_modules):
        logger.debug(
            "Deferring rlrmp recipe registration; an rlrmp.train module is still initializing."
        )
        return

    # Mark in-progress BEFORE the heavy imports below: importing
    # ``rlrmp.analysis.*`` re-enters ``rlrmp.analysis.__init__``, which calls
    # back into this function. The sentinel breaks that re-entrant cycle.
    _RECIPES_REGISTERED = "in_progress"
    try:
        from rlrmp.runtime.spec_migrations import ensure_rlrmp_spec_families
        from rlrmp.eval.recipes import register_rlrmp_evaluation_recipes
        from rlrmp.analysis.matrix import register_standard_matrix_recipes
        from rlrmp.analysis.declarative_materialization import (
            register_declarative_materialization_recipes,
        )
        from rlrmp.analysis.training_diagnostics import (
            register_training_diagnostics_recipes,
        )

        ensure_rlrmp_spec_families()
        register_rlrmp_evaluation_recipes(replace=True)
        register_standard_matrix_recipes()
        register_declarative_materialization_recipes(replace=True)
        register_training_diagnostics_recipes()
    except Exception:
        # Reset so a later call re-attempts registration.
        _RECIPES_REGISTERED = None
        raise

    _RECIPES_REGISTERED = True
