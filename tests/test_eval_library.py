"""Smoke tests for `rlrmp.eval` and `rlrmp.train` library modules.

Bug: 8404108 — these guard the extracted primitives that used to live in
``scripts/eval_part2_5_figures.py`` and the two trainer scripts. They are
deliberately import-and-shape level tests; full numerical tests of the
underlying eval pipeline are covered by the experiment-side analyses.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# rlrmp.eval public surface
# ---------------------------------------------------------------------------

def test_rlrmp_eval_public_surface():
    """The `rlrmp.eval` package re-exports its six public names."""
    import rlrmp.eval as ev
    for name in (
        "N_REPLICATES",
        "compute_kinematics",
        "eval_at_pert0",
        "eval_at_pert_scale",
        "eval_ensemble_on_trials",
        "set_sisu",
    ):
        assert hasattr(ev, name), f"rlrmp.eval missing {name}"


def test_rlrmp_eval_minimax_io_loaders_importable():
    """`rlrmp.eval.minimax_io` provides three loaders."""
    from rlrmp.eval import minimax_io
    assert callable(minimax_io.load_config)
    assert callable(minimax_io.load_model)
    assert callable(minimax_io.load_adversary)


def test_n_replicates_constant():
    """`N_REPLICATES` is the project-wide default of 5."""
    from rlrmp.eval.ensemble import N_REPLICATES
    assert N_REPLICATES == 5


# ---------------------------------------------------------------------------
# rlrmp.train public surface
# ---------------------------------------------------------------------------

def test_rlrmp_train_public_surface():
    """The `rlrmp.train` package re-exports both build_hps_* helpers."""
    import rlrmp.train as tr
    assert callable(tr.build_hps_minimax)
    assert callable(tr.build_hps_standard)


def test_build_hps_minimax_signature_matches():
    """`rlrmp.train.minimax.build_hps` accepts an argparse Namespace with the
    documented minimum field set, and returns a TreeNamespace with the same
    top-level structure as the legacy in-script `build_hps`.
    """
    from feedbax.config.namespace import TreeNamespace
    from rlrmp.train.minimax import build_hps

    args = argparse.Namespace(
        n_warmup_batches=10,
        n_adversary_batches=20,
        controller_lr=0.01,
        loss_update_enabled=False,
        loss_update_ratio=0.3,
        hidden_type="gru",
        sisu_gating="additive",
        # Optional fields filled with defaults via getattr() inside build_hps:
    )
    hps = build_hps(args)
    assert isinstance(hps, TreeNamespace)
    for top in ("method", "dt", "model", "task", "pert", "loss", "loss_update",
                "where", "hidden_type", "sisu_gating"):
        assert hasattr(hps, top), f"hps missing top-level key {top}"


def test_build_hps_standard_signature_matches():
    """`rlrmp.train.standard.build_hps` accepts the documented loss-mode menu."""
    from feedbax.config.namespace import TreeNamespace
    from rlrmp.train.standard import LOSS_MODE_CONFIGS, build_hps

    # All four loss modes should be present.
    assert set(LOSS_MODE_CONFIGS.keys()) == {
        "running_cost", "softmin", "combined", "default",
    }

    args = argparse.Namespace(
        n_batches=100,
        pert_std=1.0,
        loss_mode="running_cost",
        target_ratio=0.3,
        enable_loss_update=False,
        nn_output=1e-5,
    )
    hps = build_hps(args)
    assert isinstance(hps, TreeNamespace)
    assert hasattr(hps, "loss")
    # nn_output override should be reflected
    assert float(hps.loss.weights.nn_output) == pytest.approx(1e-5)


# ---------------------------------------------------------------------------
# rlrmp.eval.kinematics — exercise the no-replicate-dim branch on stub data
# ---------------------------------------------------------------------------

class _Effector:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel


class _Mechanics:
    def __init__(self, effector):
        self.effector = effector


class _States:
    def __init__(self, mechanics):
        self.mechanics = mechanics


class _Timeline:
    def __init__(self, epoch_bounds):
        self.epoch_bounds = epoch_bounds


class _TargetValue:
    def __init__(self, value):
        self.value = value


class _TrialSpecs:
    def __init__(self, targets, timeline):
        self.targets = targets
        self.timeline = timeline


def test_compute_kinematics_shapes_no_replicate_dim():
    """compute_kinematics returns three (n_trials,) arrays without a rep axis."""
    from rlrmp.eval.kinematics import compute_kinematics

    n_trials, n_steps = 3, 8
    # Linear motion in x toward goal at (1, 0)
    pos = jnp.zeros((n_trials, n_steps, 2)).at[:, :, 0].set(
        jnp.linspace(0.0, 1.0, n_steps)[None, :].repeat(n_trials, axis=0)
    )
    vel = jnp.zeros((n_trials, n_steps, 2)).at[:, :, 0].set(1.0)
    states = _States(_Mechanics(_Effector(pos, vel)))
    # Single target shaped (n_trials, n_steps, 2); final-step is the goal.
    target_value = jnp.tile(
        jnp.array([1.0, 0.0])[None, None, :], (n_trials, n_steps, 1),
    )
    trial_specs = _TrialSpecs(
        targets={"target": _TargetValue(target_value)},
        timeline=_Timeline(
            epoch_bounds=jnp.tile(jnp.array([0, 1, 2, n_steps])[None, :], (n_trials, 1))
        ),
    )

    out = compute_kinematics(states, trial_specs)
    assert set(out.keys()) == {"peak_velocity", "endpoint_error", "max_lateral_deviation"}
    for k, v in out.items():
        assert isinstance(v, np.ndarray)
        assert v.shape == (n_trials,), f"{k}: expected ({n_trials},) got {v.shape}"
