"""Regression tests for rlrmp's lazy feedbax plugin registration (issue 462bb31).

Plugin discovery is triggered DURING ``feedbax.__init__`` (``feedbax.config``
accesses the experiment registry before ``feedbax``'s public API — e.g.
``feedbax.AbstractTask`` — is defined). rlrmp's ``register_experiment_package``
therefore must stay lightweight: register only package metadata, and defer the
heavy recipe registration (which imports ``feedbax.analysis.*`` /
``feedbax.contracts.*``) until first real use of an ``rlrmp.analysis`` module,
which only happens after ``feedbax`` has finished initializing.

The eager version silently dropped the rlrmp plugin with::

    Failed to load experiment package 'rlrmp': cannot import name 'AbstractTask'
    from partially initialized module 'feedbax'

leaving the experiment registry empty.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import rlrmp
from feedbax.plugins.registry import ExperimentRegistry


def test_register_experiment_package_defers_recipes_while_feedbax_initializing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While ``feedbax`` is still mid-init, ``register_experiment_package``
    registers only package metadata and defers the heavy recipe registration.

    This is the registration-time invariant that keeps mid-init plugin discovery
    safe: the call must not depend on the full feedbax/rlrmp public API being
    importable, and must not import (and prematurely cache) ``rlrmp.analysis``.
    """
    # Simulate the mid-init condition that triggers the bug: feedbax's public
    # API is not ready yet.
    monkeypatch.setattr(rlrmp, "_RECIPES_REGISTERED", None)
    monkeypatch.setattr(rlrmp, "_feedbax_public_api_ready", lambda: False)

    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    # Metadata registered (registry non-empty) ...
    assert registry.get_package_names() == ["rlrmp"]
    assert registry.single_package_name() == "rlrmp"

    # ... but the heavy recipe registration is deferred, not run.
    assert rlrmp._RECIPES_REGISTERED is None


def test_register_experiment_package_registers_recipes_when_feedbax_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``feedbax`` is ready, ``register_experiment_package`` registers
    rlrmp's recipes, and re-invoking it restores them even after they were
    swapped out of feedbax's process-global recipe registry."""
    from feedbax.analysis.specs import (
        registered_analysis_types,
        unregister_analysis_recipe,
    )

    monkeypatch.setattr(rlrmp, "_RECIPES_REGISTERED", None)

    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    assert rlrmp._RECIPES_REGISTERED is True
    rlrmp_recipes = [r for r in registered_analysis_types() if r.startswith("rlrmp.")]
    assert rlrmp_recipes

    # Another caller swaps an rlrmp recipe out of feedbax's recipe registry.
    victim = rlrmp_recipes[0]
    unregister_analysis_recipe(victim)
    assert victim not in registered_analysis_types()

    # Re-invoking register_experiment_package restores it (idempotent force).
    rlrmp.register_experiment_package(registry)
    assert victim in registered_analysis_types()


def test_ensure_rlrmp_recipes_registered_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recipe registration runs once, populates feedbax's recipe registry, and
    is a no-op on subsequent calls."""
    from feedbax.analysis.specs import registered_analysis_types

    # Force a fresh run even if recipes were already registered by an earlier
    # import in this process.
    monkeypatch.setattr(rlrmp, "_RECIPES_REGISTERED", None)

    rlrmp.ensure_rlrmp_recipes_registered()
    assert rlrmp._RECIPES_REGISTERED is True

    recipes = set(registered_analysis_types())
    assert any(name.startswith("rlrmp.") for name in recipes), sorted(recipes)

    # Idempotent: a second call is a no-op and leaves the flag set.
    rlrmp.ensure_rlrmp_recipes_registered()
    assert rlrmp._RECIPES_REGISTERED is True


def test_importing_rlrmp_analysis_registers_recipes() -> None:
    """In a fresh process, importing ``rlrmp.analysis`` (the deferred-trigger
    package init) registers rlrmp's recipes — without any explicit
    ``register_experiment_package`` or ``ensure_rlrmp_recipes_registered`` call.

    This is the deferred-registration trigger that fires post-``feedbax``-init
    in real downstream use, recovering the recipes that mid-init discovery had
    to skip. A subprocess is used because the trigger is an import-time side
    effect of ``rlrmp.analysis.__init__``, which is cached once per process.
    """
    script = textwrap.dedent(
        """
        import feedbax  # noqa: F401  -- mid-init discovery defers rlrmp recipes
        from feedbax.analysis.specs import registered_analysis_types

        import rlrmp
        assert rlrmp._RECIPES_REGISTERED is None, "recipes should start deferred"

        import rlrmp.analysis  # noqa: F401  -- import-time trigger

        assert rlrmp._RECIPES_REGISTERED is True, rlrmp._RECIPES_REGISTERED
        recipes = [r for r in registered_analysis_types() if r.startswith("rlrmp.")]
        assert recipes, "rlrmp recipes not registered by analysis import"
        print("RECIPES_OK:" + ",".join(sorted(recipes)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing rlrmp.analysis did not register recipes.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "RECIPES_OK:" in result.stdout


def test_fresh_import_feedbax_registers_rlrmp_cleanly() -> None:
    """A fresh ``import feedbax`` in a clean subprocess registers the rlrmp
    experiment package without the partially-initialized-``feedbax`` cycle and
    leaves the registry non-empty.

    This is the end-to-end regression for issue 462bb31. It treats any
    ``feedbax.plugins`` discovery warning as a failure, which is exactly the
    silent-drop signature of the original bug.
    """
    script = textwrap.dedent(
        """
        import logging
        import sys

        class _FailOnDiscoveryWarning(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING and "plugins" in record.name:
                    sys.stderr.write("DISCOVERY_WARNING:" + record.getMessage() + "\\n")
                    sys.exit(3)

        logging.getLogger("feedbax.plugins").addHandler(_FailOnDiscoveryWarning())

        import feedbax  # noqa: F401  -- triggers mid-init plugin discovery
        from feedbax.plugins import EXPERIMENT_REGISTRY

        names = EXPERIMENT_REGISTRY.get_package_names()
        assert "rlrmp" in names, f"rlrmp plugin not registered; registry={names!r}"
        print("REGISTRY_OK:" + ",".join(names))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "fresh `import feedbax` failed to register rlrmp cleanly.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "REGISTRY_OK:" in result.stdout
    assert "rlrmp" in result.stdout
