"""Spot-check mult_single replicates 0–4 for induced-gain outlier diagnosis.

Bug: 4f2e934 — determine whether replicate 0's γ_af = 15.6 outlier is
replicate-specific or method-wide.

Runs additive_force × qr_cost only (the headline discriminator channel).
Uses the same analyser setup as run_induced_gain_part2_5.py.

Usage:
    cd /Users/mll/Main/10\ Projects/10\ PhD/rlrmp/worktrees/feature__mult-single-replicate-check
    uv run python scripts/spot_check_mult_single.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np

from rlrmp.analysis.math.hinf_riccati import (
    CostSpec,
    cost_schedule_from_spec,
    find_gamma_star,
    linearize_pointmass,
)
from rlrmp.analysis.math.induced_gain import (
    W_ADDITIVE_FORCE,
    Z_QR_COST,
    induced_gain,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Import helpers from the main runner (sibling module in the same directory).
from run_induced_gain_part2_5 import (
    GroupSpec,
    analyse_group,
    load_group_model,
    build_network_controller,
)

# Repo root, used to construct paths into _artifacts/. Bug: 8404108 — switched
# from a manual __file__-relative computation to the canonical rlrmp.paths.REPO_ROOT
# so this script works regardless of its location in the repo (top-level scripts/
# or results/<hash>/scripts/).
from rlrmp.paths import REPO_ROOT as WORKTREE  # noqa: E402


RUNPOD_ROOT = Path("/Users/mll/Main/10 Projects/10 PhD/rlrmp/_artifacts/part2_5/runpod")
MULT_SINGLE_SPEC = GroupSpec(
    "mult_single",
    "mult_single",
    "train_minimax",
    "adversarial",
)

N_REPLICATES = 5
HORIZON = 200
INIT_POS = jnp.array([0.0, 0.0], dtype=jnp.float64)
TARGET_POS = jnp.array([0.15, 0.0], dtype=jnp.float64)
SISU = 0.5
SEED = 0
N_RESTARTS = 3
MAX_ITER = 600
RTOL = 1e-5


def main():
    plant = linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)
    spec = CostSpec(n_steps=HORIZON)
    schedule = cost_schedule_from_spec(spec, plant)

    logger.info("Computing Riccati γ⋆ baseline...")
    g_star = find_gamma_star(plant, schedule)
    logger.info("γ⋆ = %.6f", g_star)

    channels = ((W_ADDITIVE_FORCE, Z_QR_COST),)

    results = {}
    for rep_idx in range(N_REPLICATES):
        logger.info("--- Replicate %d ---", rep_idx)
        t_start = time.time()
        try:
            model, task, config = load_group_model(
                MULT_SINGLE_SPEC, RUNPOD_ROOT, replicate_idx=rep_idx
            )
        except Exception as e:
            logger.error("Load failed for replicate %d: %s", rep_idx, e)
            results[rep_idx] = {"error": f"load failed: {e}"}
            continue

        try:
            ctrl = build_network_controller(
                model, target_pos=TARGET_POS, sisu=SISU, key=jr.PRNGKey(SEED)
            )
        except Exception as e:
            logger.error("Adapter build failed for replicate %d: %s", rep_idx, e)
            results[rep_idx] = {"error": f"adapter build failed: {e}"}
            continue

        logger.info("  n_ctrl=%d  delay=%d", ctrl.h0_flat.shape[0], ctrl.delay)

        try:
            res = induced_gain(
                plant,
                ctrl,
                init_pos=INIT_POS,
                target_pos=TARGET_POS,
                horizon=HORIZON,
                w_channel=W_ADDITIVE_FORCE,
                z_channel=Z_QR_COST,
                schedule=schedule,
                methods=("power_iteration",),
                n_restarts=N_RESTARTS,
                max_iter=MAX_ITER,
                rtol=RTOL,
                seed=SEED,
            )
            pi = res["power_iteration"]
            elapsed = time.time() - t_start
            logger.info(
                "  replicate %d: γ_af = %.4f  (converged=%s, iters=%d, ratio/γ⋆=%.3f, %.1fs)",
                rep_idx, pi.gamma, pi.converged, pi.iterations,
                pi.gamma / g_star if g_star > 0 else float("nan"),
                elapsed,
            )
            results[rep_idx] = {
                "gamma_af": float(pi.gamma),
                "converged": bool(pi.converged),
                "iterations": int(pi.iterations),
                "ratio_to_gstar": float(pi.gamma / g_star) if g_star > 0 else float("nan"),
                "elapsed_s": elapsed,
            }
        except Exception as e:
            logger.error("  Analyser failed for replicate %d: %s", rep_idx, e)
            results[rep_idx] = {"error": f"analyser failed: {e}"}

    # Summary table
    print("\n=== mult_single replicate spot-check ===")
    print(f"γ⋆ (Riccati) = {g_star:.6f}")
    print(f"{'Rep':>4}  {'γ_af':>10}  {'γ/γ⋆':>10}  {'conv':>6}  {'iters':>6}")
    print("-" * 50)
    for rep_idx in range(N_REPLICATES):
        r = results.get(rep_idx, {})
        if "error" in r:
            print(f"{rep_idx:>4}  {'ERROR':>10}  {'—':>10}  {'—':>6}  {'—':>6}  {r['error']}")
        else:
            conv_mark = "Y" if r.get("converged") else "N"
            print(
                f"{rep_idx:>4}  {r['gamma_af']:>10.4f}  {r['ratio_to_gstar']:>10.3f}"
                f"  {conv_mark:>6}  {r['iterations']:>6}"
            )

    # Save results JSON to _artifacts for reference.
    out_dir = WORKTREE / "_artifacts" / "part2_5" / "runs" / "mult_single_replicate_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "replicate_gains.json"
    with open(out_path, "w") as f:
        json.dump({
            "gamma_star_riccati": float(g_star),
            "group": "mult_single",
            "replicates": {str(k): v for k, v in results.items()},
        }, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
