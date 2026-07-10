# ruff: noqa: E402
"""Null-space decomposition of SISU effects on converged center-out models.

For each model, runs the forward pass at SISU=0 and SISU=1 (pert_scale=0),
extracts hidden state trajectories h(t), computes delta_h = h(SISU=1) - h(SISU=0),
and projects onto the output-potent and output-null subspaces of the readout matrix.

Also performs population-specific analysis (input, readout, recurrent populations).

Usage:
    uv run python results/2ef67ca/scripts/eval_null_space.py
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import jax.random as jr
import numpy as np
from jax_cookbook import load_with_hyperparameters

from rlrmp.eval.pert import eval_states_at_pert_scale
from rlrmp.train.standard import build_hps
from rlrmp.train.task_model import setup_task_model_pair

RESULTS_BASE = Path(__file__).resolve().parent.parent
# ---------------------------------------------------------------------------
# Paths — archived models live beside these legacy scripts.
# ---------------------------------------------------------------------------

MODELS_BASE = RESULTS_BASE / "models"


def _find_model_dir(model_dir_name: str) -> Path | None:
    """Find an archived model directory."""
    candidate = MODELS_BASE / model_dir_name
    if (candidate / "trained_model.eqx").exists():
        return candidate
    return None


# Models to analyze
CONDITIONS = [
    "centerout_baseline_std_pert0",
    "centerout_baseline_std_pert1",
    "centerout_std_update_r03_pert1",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_condition(model_dir_name: str):
    """Load task and trained model for a condition.

    Returns:
        Tuple of (task, trained_model, config), or None if not found.
    """
    config_path = MODELS_BASE / model_dir_name / "config.json"
    if not config_path.exists():
        return None

    cond_dir = _find_model_dir(model_dir_name)
    if cond_dir is None:
        return None

    with open(config_path) as f:
        config = json.load(f)

    args = argparse.Namespace(**config)
    hps = build_hps(args)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)

    trained_model, _ = load_with_hyperparameters(
        cond_dir / "trained_model.eqx",
        setup_func=lambda key, **kwargs: setup_task_model_pair(hps, key=key).model,
    )

    return pair.task, trained_model, config


# ---------------------------------------------------------------------------
# Null-space decomposition helpers
# ---------------------------------------------------------------------------


def get_readout_weight(model) -> np.ndarray:
    """Get effective readout weight matrix for one replicate.

    The MaskedLinear has shape (n_replicates, out_size, hidden_size).
    We average over replicates after masking.

    Args:
        model: Trained ensemble model (SimpleFeedback).

    Returns:
        W: (out_size, hidden_size) effective readout weight matrix, averaged over replicates.
    """
    readout = model.net.readout  # MaskedLinear
    W_ensemble = readout.linear.weight * readout.mask  # (n_rep, out_size, hidden_size)
    return np.array(W_ensemble)  # (n_rep, out_size, hidden_size)


def compute_potent_null_projections(W: np.ndarray):
    """Compute output-potent and output-null projection matrices.

    Args:
        W: (out_size, hidden_size) readout weight matrix.

    Returns:
        P_potent: (hidden_size, hidden_size) projection onto potent subspace.
        P_null: (hidden_size, hidden_size) projection onto null subspace.
        rank: Estimated rank of W.
    """
    # SVD of W.T: columns of U are the potent directions in hidden space
    # W.T has shape (hidden_size, out_size)
    U, S, Vt = np.linalg.svd(W.T, full_matrices=False)  # U: (hidden, min(h,o))
    rank = int(np.sum(S > 1e-6 * S[0]))
    U_potent = U[:, :rank]  # (hidden, rank) — at most out_size columns
    P_potent = U_potent @ U_potent.T  # (hidden, hidden)
    P_null = np.eye(W.shape[1]) - P_potent  # (hidden, hidden)
    return P_potent, P_null, rank


def eval_hidden_states(task, model, sisu: float, *, key) -> np.ndarray:
    """Run forward pass at given SISU with pert_scale=0.

    Args:
        task: Task object.
        model: Ensemble model.
        sisu: SISU level.
        key: Random key.

    Returns:
        h: (n_replicates, n_trials, n_timesteps, hidden_size) hidden states.
    """
    states, _ = eval_states_at_pert_scale(
        task,
        model,
        sisu,
        0.0,
        key=key,
    )
    return np.array(states.net.hidden)  # (n_rep, n_trials, n_timesteps, hidden_size)


def null_space_analysis(
    h_s0: np.ndarray,
    h_s1: np.ndarray,
    P_potent: np.ndarray,
    P_null: np.ndarray,
) -> dict:
    """Compute potent/null decomposition of delta_h.

    Args:
        h_s0: (n_rep, n_trials, n_timesteps, hidden_size) hidden states at SISU=0.
        h_s1: (n_rep, n_trials, n_timesteps, hidden_size) hidden states at SISU=1.
        P_potent: (hidden_size, hidden_size) potent projection.
        P_null: (hidden_size, hidden_size) null projection.

    Returns:
        Dict with:
            - delta_h: (n_rep, n_trials, n_timesteps, hidden_size)
            - delta_h_potent: same shape, potent component
            - delta_h_null: same shape, null component
            - potent_norm: (n_rep, n_trials, n_timesteps) L2 norm of potent component
            - null_norm: (n_rep, n_trials, n_timesteps) L2 norm of null component
            - potent_frac: scalar, fraction of variance in potent subspace (time-averaged)
            - null_frac: scalar, fraction of variance in null subspace (time-averaged)
    """
    delta_h = h_s1 - h_s0  # (n_rep, n_trials, n_timesteps, hidden_size)

    # Project: delta_h @ P.T = delta_h @ P (P is symmetric)
    # Broadcasting: (..., hidden) @ (hidden, hidden) -> (..., hidden)
    delta_h_potent = delta_h @ P_potent  # (n_rep, n_trials, n_timesteps, hidden_size)
    delta_h_null = delta_h @ P_null

    potent_norm = np.linalg.norm(delta_h_potent, axis=-1)  # (n_rep, n_trials, n_timesteps)
    null_norm = np.linalg.norm(delta_h_null, axis=-1)
    total_norm = np.linalg.norm(delta_h, axis=-1)

    # Fraction: sum of squared norms across trials, then average over time
    potent_sq = np.sum(potent_norm**2, axis=(0, 1))  # (n_timesteps,)
    null_sq = np.sum(null_norm**2, axis=(0, 1))
    total_sq = np.sum(total_norm**2, axis=(0, 1))

    potent_frac_by_time = np.where(total_sq > 0, potent_sq / total_sq, 0.0)
    null_frac_by_time = np.where(total_sq > 0, null_sq / total_sq, 0.0)

    potent_frac = float(np.mean(potent_frac_by_time))
    null_frac = float(np.mean(null_frac_by_time))

    return {
        "delta_h": delta_h,
        "delta_h_potent": delta_h_potent,
        "delta_h_null": delta_h_null,
        "potent_norm": potent_norm,
        "null_norm": null_norm,
        "total_norm": total_norm,
        "potent_frac": potent_frac,
        "null_frac": null_frac,
        "potent_frac_by_time": potent_frac_by_time,
        "null_frac_by_time": null_frac_by_time,
    }


def population_analysis(
    delta_h: np.ndarray,
    pop_struct,
    rep_idx: int = 0,
) -> dict:
    """Compute norm of delta_h within each population subspace.

    Args:
        delta_h: (n_rep, n_trials, n_timesteps, hidden_size) SISU effect.
        pop_struct: PopulationStructure from model.net.population_structure.
        rep_idx: Which replicate's population indices to use (default 0).

    Returns:
        Dict with per-population mean norms (averaged over replicates, trials, time).
    """
    # Population indices have shape (n_rep, n_pop_units)
    input_idx = np.array(pop_struct.input_only_indices)  # (n_rep, n_input_only)
    readout_idx = np.array(pop_struct.readout_only_indices)
    recurrent_idx = np.array(pop_struct.recurrent_only_indices)

    results = {}
    n_rep = delta_h.shape[0]

    for pop_name, idx_arr in [
        ("input", input_idx),
        ("readout", readout_idx),
        ("recurrent", recurrent_idx),
    ]:
        # Average norm across replicates using each replicate's own indices
        norms_per_rep = []
        for r in range(n_rep):
            idx = idx_arr[r]  # (n_units,)
            dh_pop = delta_h[r, :, :, idx]  # (n_trials, n_timesteps, n_units)
            norm_pop = np.linalg.norm(dh_pop, axis=-1)  # (n_trials, n_timesteps)
            norms_per_rep.append(float(np.mean(norm_pop)))
        results[f"{pop_name}_norm"] = float(np.mean(norms_per_rep))
        results[f"{pop_name}_norm_std"] = float(np.std(norms_per_rep))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading models...\n")
    loaded = {}
    for model_dir in CONDITIONS:
        print(f"  {model_dir} ... ", end="", flush=True)
        result = load_condition(model_dir)
        if result is None:
            print("SKIPPED — not found")
            loaded[model_dir] = None
        else:
            print("OK")
            loaded[model_dir] = result
    print()

    # =========================================================================
    # Null-space decomposition
    # =========================================================================
    print("=" * 100)
    print("NULL-SPACE DECOMPOSITION OF SISU EFFECTS (pert_scale=0)")
    print("=" * 100)
    print()

    results = {}
    for model_dir in CONDITIONS:
        entry = loaded[model_dir]
        if entry is None:
            results[model_dir] = None
            continue
        task, model, config = entry
        print(f"  {model_dir}: evaluating at SISU=0 and SISU=1 ...")

        key = jr.PRNGKey(200)
        k0, k1 = jr.split(key)
        h_s0 = eval_hidden_states(task, model, sisu=0.0, key=k0)
        h_s1 = eval_hidden_states(task, model, sisu=1.0, key=k1)

        # Use replicate-averaged readout weights for the projection
        W_ensemble = get_readout_weight(model)  # (n_rep, out_size, hidden_size)
        W_mean = W_ensemble.mean(axis=0)  # (out_size, hidden_size)

        P_potent, P_null, rank = compute_potent_null_projections(W_mean)

        decomp = null_space_analysis(h_s0, h_s1, P_potent, P_null)
        pop_result = population_analysis(decomp["delta_h"], model.net.population_structure)

        results[model_dir] = {
            "decomp": decomp,
            "pop": pop_result,
            "rank": rank,
            "W_shape": W_mean.shape,
            "hidden_shape": h_s0.shape,
        }
        print(f"    rank={rank}, W={W_mean.shape}, h={h_s0.shape}")
    print()

    # =========================================================================
    # Table 1: potent vs null fractions
    # =========================================================================
    print("=" * 100)
    print("TABLE 1: Potent vs. null-space decomposition of SISU effect")
    print("=" * 100)
    print()
    col_w = 35
    hdr = f"{'Model':<{col_w}} | {'potent_frac':>12} | {'null_frac':>10} | {'potent/null':>12} | {'rank':>5}"
    print(hdr)
    print("-" * len(hdr))
    for model_dir in CONDITIONS:
        r = results[model_dir]
        if r is None:
            print(f"{'  ' + model_dir:<{col_w}} | {'MISSING':>12}")
            continue
        d = r["decomp"]
        pf = d["potent_frac"]
        nf = d["null_frac"]
        ratio = pf / nf if nf > 1e-12 else float("inf")
        print(
            f"{model_dir:<{col_w}} | {pf:>12.4f} | {nf:>10.4f} | {ratio:>12.4f} | {r['rank']:>5}"
        )
    print()

    # =========================================================================
    # Table 2: time course of potent/null norms (10 evenly-spaced timesteps)
    # =========================================================================
    print("=" * 100)
    print("TABLE 2: Time course of potent and null norms of delta_h")
    print("(norms averaged over replicates and trials; 10 evenly-spaced timesteps)")
    print("=" * 100)
    print()

    for model_dir in CONDITIONS:
        r = results[model_dir]
        if r is None:
            print(f"  {model_dir}: MISSING")
            continue
        d = r["decomp"]
        n_timesteps = d["potent_norm"].shape[-1]
        step_indices = np.linspace(0, n_timesteps - 1, 10, dtype=int)

        # Average over replicates and trials
        potent_mean = d["potent_norm"].mean(axis=(0, 1))  # (n_timesteps,)
        null_mean = d["null_norm"].mean(axis=(0, 1))
        total_mean = d["total_norm"].mean(axis=(0, 1))

        print(f"  {model_dir} (n_timesteps={n_timesteps}):")
        hdr2 = f"    {'t':>6} | {'potent_norm':>12} | {'null_norm':>10} | {'total_norm':>12} | {'potent_frac':>12}"
        print(hdr2)
        print("    " + "-" * (len(hdr2) - 4))
        for t in step_indices:
            pn = potent_mean[t]
            nn = null_mean[t]
            tn = total_mean[t]
            pfrac = pn / tn if tn > 1e-12 else 0.0
            print(f"    {t:>6} | {pn:>12.4f} | {nn:>10.4f} | {tn:>12.4f} | {pfrac:>12.4f}")
        print()

    # =========================================================================
    # Table 3: population-specific norms
    # =========================================================================
    print("=" * 100)
    print("TABLE 3: Population-specific norms of SISU effect (delta_h)")
    print("(mean over replicates, trials, and time; ±std over replicates)")
    print("=" * 100)
    print()
    hdr3 = (
        f"{'Model':<{col_w}} | {'input_norm':>12} | {'readout_norm':>13} | {'recurrent_norm':>15}"
    )
    print(hdr3)
    print("-" * len(hdr3))
    for model_dir in CONDITIONS:
        r = results[model_dir]
        if r is None:
            print(f"{'  ' + model_dir:<{col_w}} | {'MISSING':>12}")
            continue
        p = r["pop"]
        print(
            f"{model_dir:<{col_w}} | "
            f"{p['input_norm']:>9.4f}±{p['input_norm_std']:.3f} | "
            f"{p['readout_norm']:>10.4f}±{p['readout_norm_std']:.3f} | "
            f"{p['recurrent_norm']:>12.4f}±{p['recurrent_norm_std']:.3f}"
        )
    print()

    print("Notes:")
    print("  - All evaluations use pert_scale=0 (perturbation amplitude zeroed).")
    print("  - Potent subspace: row space of the readout weight matrix W (out_size x hidden_size).")
    print("  - Null subspace: orthogonal complement of the potent subspace in hidden space.")
    print("  - potent_frac and null_frac are computed as fraction of total squared norm,")
    print("    averaged over time.")
    print("  - Population norms are averaged over replicates, trials, and timesteps.")
    print("  - ±std is across 5 replicates (each using its own population indices).")


if __name__ == "__main__":
    main()
