"""Comprehensive diagnostics for minimax training experiments.

Reports three categories of information:

1. SISU weight norms — verifies that the trained network's SISU input channel
   has learned near-zero (additive) or appropriate (multiplicative) weights by
   comparing hidden-state sensitivity to SISU vs. other inputs.

2. Comprehensive evaluation metrics — per-replicate statistics across
   SISU levels and perturbation conditions, including peak velocity, endpoint
   error, lateral deviation, and aggregate robustening effect.

3. Adversary force profile plot — time-series of adversary force magnitudes,
   including across replicates and population-adversary members.

Expected directory layout (produced by train_minimax.py):
    config.json
    warmup_model.eqx
    adversarial_model.eqx
    trained_adversary.eqx
    adversary_force_profiles.npz   (optional)

Usage:
    uv run python scripts/eval_diagnostics.py --results-dir results/tier1b/mult_single
    uv run python scripts/eval_diagnostics.py --results-dir results/myrun --pert-scale 3.0
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import matplotlib.pyplot as plt
import numpy as np

WORKTREE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKTREE / "scripts"))

from train_minimax import build_hps  # noqa: E402
from eval_minimax import load_config, load_model, load_adversary  # noqa: E402
from eval_part2_5_figures import (  # noqa: E402
    eval_ensemble_on_trials,
    compute_kinematics,
    set_sisu,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL  # noqa: E402
from rlrmp.modules.training.part2 import setup_task_model_pair  # noqa: E402

N_REPLICATES = 5


# ---------------------------------------------------------------------------
# SISU weight diagnostics
# ---------------------------------------------------------------------------


def _is_replicate_batched(x: object, n_rep: int) -> bool:
    """Return True if x is a JAX array with a leading replicate dimension."""
    return eqx.is_array(x) and hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == n_rep


def _get_single_replicate(model, rep_idx: int = 0):
    """Extract a single-replicate model from a stacked ensemble.

    Args:
        model: Ensemble model with leading replicate axis on array leaves.
        rep_idx: Which replicate to extract (default 0).

    Returns:
        A single-model PyTree with the replicate axis removed.
    """
    n_rep = _count_replicates(model)

    def _is_batched(x):
        return _is_replicate_batched(x, n_rep)

    return jt.map(
        lambda x: x[rep_idx] if _is_batched(x) else x,
        model,
        is_leaf=eqx.is_array,
    )


def _count_replicates(model) -> int:
    """Return the replicate count inferred from model array leaves."""
    leaves = eqx.filter(model, eqx.is_array)
    shapes = [x.shape for x in jt.leaves(leaves, is_leaf=eqx.is_array) if eqx.is_array(x)]
    for s in shapes:
        if len(s) >= 1 and 1 < s[0] <= 20:
            return s[0]
    return 1


def check_sisu_weights(model, config: dict) -> dict:
    """Check SISU weight norms to verify functional weight suppression.

    Uses a functional approach: runs a single GRU/RNN step with SISU=0 and
    SISU=1 to measure the resulting hidden-state sensitivity.  Also checks
    the sisu_alpha parameter directly for multiplicative gating.

    For additive GRU/RNN: the hidden-state diff (SISU=0 vs SISU=1) should be
    near zero if the network has learned to ignore SISU.  For multiplicative
    gating: sisu_alpha norm is the key diagnostic.

    Args:
        model: The trained model.  May have a leading replicate axis.
        config: config.json dict (used to determine hidden_type and sisu_gating).

    Returns:
        Dict with diagnostic values and a summary string.
    """
    hidden_type = config.get("hidden_type", "gru")
    sisu_gating = config.get("sisu_gating", "additive")

    # Extract a single-replicate model for the weight probe
    n_rep = _count_replicates(model)
    single = _get_single_replicate(model, rep_idx=0)

    net = single.nodes["net"]
    hidden_cell = net.hidden

    results: dict = {
        "hidden_type": hidden_type,
        "sisu_gating": sisu_gating,
    }

    # 1. Multiplicative-gating: check sisu_alpha norm directly
    if sisu_gating == "multiplicative":
        sisu_alpha = getattr(net, "sisu_alpha", None)
        if sisu_alpha is not None:
            alpha_norm = float(jnp.linalg.norm(sisu_alpha))
            alpha_mean_abs = float(jnp.mean(jnp.abs(sisu_alpha)))
            alpha_max_abs = float(jnp.max(jnp.abs(sisu_alpha)))
            results["sisu_alpha_norm"] = alpha_norm
            results["sisu_alpha_mean_abs"] = alpha_mean_abs
            results["sisu_alpha_max_abs"] = alpha_max_abs
        else:
            results["sisu_alpha_norm"] = None

    # 2. Functional probe: hidden-state diff between SISU=0 and SISU=1
    # We run one GRU/RNN step from a zero hidden state with a zero input except
    # for the SISU channel, which we set to 0 vs 1.
    try:
        if hidden_type == "gru":
            input_size = hidden_cell.weight_ih.shape[1]
        elif hidden_type == "vanilla_rnn":
            input_size = hidden_cell._cell.weight_ih.shape[1]
        else:
            input_size = None

        if input_size is not None:
            hidden_size = net.hidden_size if hasattr(net, "hidden_size") else (
                hidden_cell.weight_ih.shape[0] // 3
                if hidden_type == "gru"
                else hidden_cell._cell.weight_ih.shape[0]
            )
            h0 = jnp.zeros(hidden_size)
            x_base = jnp.zeros(input_size)
            x_sisu0 = x_base.at[-1].set(0.0)
            x_sisu1 = x_base.at[-1].set(1.0)

            if hidden_type == "gru":
                h_at_0 = hidden_cell(x_sisu0, h0)
                h_at_1 = hidden_cell(x_sisu1, h0)
            else:
                h_at_0 = hidden_cell(x_sisu0, h0)
                h_at_1 = hidden_cell(x_sisu1, h0)

            diff_norm = float(jnp.linalg.norm(h_at_1 - h_at_0))
            results["hidden_diff_sisu0_vs_1"] = diff_norm
            results["input_size"] = input_size

            # Compare to a random non-SISU input channel for reference
            x_nonSISU0 = x_base.at[0].set(0.0)
            x_nonSISU1 = x_base.at[0].set(1.0)
            if hidden_type == "gru":
                h_ns0 = hidden_cell(x_nonSISU0, h0)
                h_ns1 = hidden_cell(x_nonSISU1, h0)
            else:
                h_ns0 = hidden_cell(x_nonSISU0, h0)
                h_ns1 = hidden_cell(x_nonSISU1, h0)
            nnsisu_diff = float(jnp.linalg.norm(h_ns1 - h_ns0))
            results["hidden_diff_nonSISU_input0"] = nnsisu_diff
            results["sisu_ratio"] = diff_norm / (nnsisu_diff + 1e-12)

    except Exception as exc:
        results["probe_error"] = str(exc)

    # 3. For GRU additive: weight_ih last column norms
    if sisu_gating == "additive" and hidden_type == "gru":
        try:
            w = hidden_cell.weight_ih  # (3*hidden_size, input_size)
            sisu_col = w[:, -1]       # weights for SISU channel
            other_cols = w[:, :-1]    # weights for all other inputs
            sisu_col_norm = float(jnp.linalg.norm(sisu_col))
            other_col_norms = float(jnp.linalg.norm(other_cols))
            results["weight_ih_sisu_col_norm"] = sisu_col_norm
            results["weight_ih_other_cols_norm"] = other_col_norms
            results["weight_ih_sisu_fraction"] = sisu_col_norm / (other_col_norms + 1e-12)
        except Exception as exc:
            results["weight_ih_error"] = str(exc)

    # Build a summary
    lines = []
    lines.append(f"  hidden_type:  {hidden_type}")
    lines.append(f"  sisu_gating:  {sisu_gating}")
    if sisu_gating == "multiplicative" and results.get("sisu_alpha_norm") is not None:
        lines.append(f"  sisu_alpha norm:    {results['sisu_alpha_norm']:.4e}")
        lines.append(f"  sisu_alpha mean|·|: {results['sisu_alpha_mean_abs']:.4e}")
        lines.append(f"  sisu_alpha max|·|:  {results['sisu_alpha_max_abs']:.4e}")
    if "hidden_diff_sisu0_vs_1" in results:
        lines.append(f"  Hidden-state diff SISU(0→1):          {results['hidden_diff_sisu0_vs_1']:.4e}")
        lines.append(f"  Hidden-state diff non-SISU input(0→1):{results['hidden_diff_nonSISU_input0']:.4e}")
        lines.append(f"  SISU / non-SISU sensitivity ratio:    {results['sisu_ratio']:.4e}")
        if results["sisu_ratio"] < 0.05:
            lines.append("  → Network has effectively suppressed SISU input (ratio < 0.05)")
        elif results["sisu_ratio"] < 0.2:
            lines.append("  → Network has partially suppressed SISU input")
        else:
            lines.append("  → Network is sensitive to SISU input (ratio >= 0.2)")
    if "weight_ih_sisu_col_norm" in results:
        lines.append(
            f"  weight_ih SISU col norm:   {results['weight_ih_sisu_col_norm']:.4e}"
            f"  (fraction of total: {results['weight_ih_sisu_fraction']:.4e})"
        )
    if "probe_error" in results:
        lines.append(f"  WARNING: probe failed — {results['probe_error']}")
    results["summary"] = "\n".join(lines)
    return results


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _eval_at_pert(task, model, sisu: float, pert_scale: float, *, key):
    """Evaluate model at given SISU and pert_scale.

    Args:
        task: The task object.
        model: Ensemble model with leading replicate axis.
        sisu: SISU value in [0, 1].
        pert_scale: Perturbation scale (0.0 = no perturbation).
        key: PRNG key.

    Returns:
        Dict with arrays shaped (n_replicates, n_trials):
            "peak_velocity", "endpoint_error", "max_lateral_deviation".
    """
    val_trials = task.validation_trials
    trial_specs = set_sisu(val_trials, sisu)
    pert_shape = trial_specs.intervene[PLANT_INTERVENOR_LABEL].scale.shape
    if pert_scale == 0.0:
        trial_specs = eqx.tree_at(
            lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
            trial_specs,
            jnp.zeros(pert_shape),
        )
    else:
        trial_specs = eqx.tree_at(
            lambda t: t.intervene[PLANT_INTERVENOR_LABEL].scale,
            trial_specs,
            jnp.full(pert_shape, pert_scale),
        )
    states = eval_ensemble_on_trials(task, model, trial_specs, key=key)
    return compute_kinematics(states, trial_specs)


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) over all elements."""
    return float(np.mean(arr)), float(np.std(arr))


def report_eval_metrics(
    task,
    models_dict: dict,
    pert_scale: float,
    *,
    key: jax.Array,
) -> None:
    """Print comprehensive evaluation metrics for each model.

    Metrics:
      - Peak velocity at SISU=0, 0.5, 1.0 (pert_scale=0) — mean ± std
      - Endpoint error at SISU=0.5 (pert_scale=0)
      - Peak velocity at pert_scale>0, SISU=0.5 (aggregate robustening)
      - Max lateral deviation at pert_scale>0, SISU=0.5
      - Endpoint error at pert_scale>0, SISU=0.5

    Args:
        task: Feedbax task object.
        models_dict: Mapping of model name → model PyTree.
        pert_scale: Perturbation scale for robustness evaluation.
        key: PRNG key.
    """
    SISU_LEVELS = [0.0, 0.5, 1.0]
    SISU_PERT = 0.5
    width = 14

    # -----------------------------------------------------------------------
    # Table 1: SISU velocity & endpoint error (pert_scale=0)
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("TABLE 1: SISU velocity effect (pert_scale=0)")
    print("=" * 100)

    hdr = (
        f"  {'model':>14} | {'vel(S=0)':>16} | {'vel(S=0.5)':>16} | {'vel(S=1)':>16} |"
        f" {'Δvel(0→1)':>10} | {'ep_err(S=0.5)':>16}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    t1_rows = {}
    for name, model in models_dict.items():
        row: dict = {}
        eval_key = key
        for sisu in SISU_LEVELS:
            eval_key, k = jr.split(eval_key)
            km = _eval_at_pert(task, model, sisu=sisu, pert_scale=0.0, key=k)
            mu, sd = _mean_std(km["peak_velocity"])
            mu_ep, sd_ep = _mean_std(km["endpoint_error"])
            row[f"vel_{sisu}"] = (mu, sd)
            row[f"ep_err_{sisu}"] = (mu_ep, sd_ep)
        t1_rows[name] = row

        v0_mu, v0_sd = row["vel_0.0"]
        v05_mu, v05_sd = row["vel_0.5"]
        v1_mu, v1_sd = row["vel_1.0"]
        ep_mu, ep_sd = row["ep_err_0.5"]
        delta = v1_mu - v0_mu

        def _fmt(mu, sd):
            return f"{mu:.3f} ± {sd:.3f}"

        print(
            f"  {name:>14} | {_fmt(v0_mu, v0_sd):>16} | {_fmt(v05_mu, v05_sd):>16} |"
            f" {_fmt(v1_mu, v1_sd):>16} | {delta:>+10.4f} | {_fmt(ep_mu, ep_sd):>16}"
        )

    print()

    # Commentary on SISU velocity effect
    if "warmup" in t1_rows and "adversarial" in t1_rows:
        dw = t1_rows["warmup"]["vel_1.0"][0] - t1_rows["warmup"]["vel_0.0"][0]
        da = t1_rows["adversarial"]["vel_1.0"][0] - t1_rows["adversarial"]["vel_0.0"][0]
        print(f"  Δvel (SISU 0→1) warmup: {dw:+.4f}   adversarial: {da:+.4f}")
        if da > dw + 0.005:
            print("  → Adversarial training INCREASED SISU velocity effect")
        elif da < dw - 0.005:
            print("  → Adversarial training DECREASED SISU velocity effect")
        else:
            print("  → No meaningful change in SISU velocity effect")
        print()

    # -----------------------------------------------------------------------
    # Table 2: Robustness under perturbation (pert_scale > 0)
    # -----------------------------------------------------------------------
    print("=" * 100)
    print(f"TABLE 2: Robustness under perturbation (pert_scale={pert_scale}, SISU={SISU_PERT})")
    print("=" * 100)

    # Also report unperturbed velocity for "aggregate robustening" comparison
    hdr2 = (
        f"  {'model':>14} | {'vel_unpert':>16} | {'vel_pert':>16} |"
        f" {'ep_err_pert':>16} | {'lat_dev_pert':>16}"
    )
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for name, model in models_dict.items():
        eval_key, k1, k2 = jr.split(key, 3)
        km_unp = _eval_at_pert(task, model, sisu=SISU_PERT, pert_scale=0.0, key=k1)
        km_p = _eval_at_pert(task, model, sisu=SISU_PERT, pert_scale=pert_scale, key=k2)

        vel_unp_mu, vel_unp_sd = _mean_std(km_unp["peak_velocity"])
        vel_p_mu, vel_p_sd = _mean_std(km_p["peak_velocity"])
        ep_p_mu, ep_p_sd = _mean_std(km_p["endpoint_error"])
        lat_p_mu, lat_p_sd = _mean_std(km_p["max_lateral_deviation"])

        def _fmt(mu, sd):
            return f"{mu:.3f} ± {sd:.3f}"

        print(
            f"  {name:>14} | {_fmt(vel_unp_mu, vel_unp_sd):>16} | {_fmt(vel_p_mu, vel_p_sd):>16} |"
            f" {_fmt(ep_p_mu, ep_p_sd):>16} | {_fmt(lat_p_mu, lat_p_sd):>16}"
        )

    print()
    print("  Notes:")
    print(f"    vel_unpert  = peak velocity without perturbation (SISU={SISU_PERT})")
    print(f"    vel_pert    = peak velocity under perturbation — aggregate robustening metric")
    print(f"    ep_err_pert = endpoint error under perturbation (pert_scale={pert_scale})")
    print(f"    lat_dev_pert = max lateral deviation under perturbation")
    print()


# ---------------------------------------------------------------------------
# Adversary force profile plot
# ---------------------------------------------------------------------------


def plot_adversary_force_profiles(
    results_dir: Path,
    adversaries,
    hps,
    n_replicates: int = N_REPLICATES,
) -> None:
    """Plot adversary force profiles and save to results_dir/adversary_force_profiles.png.

    Handles three cases:
      - Pre-saved adversary_force_profiles.npz: plots whatever SISU profiles are saved.
      - Single GaussianBumpAdversary: calls adversary() directly.
      - List of K adversaries (population): calls each and plots all K + mean ± std band.

    For replicate models, the adversary is not replicated — it is shared.

    Args:
        results_dir: Directory to save the plot.
        adversaries: One of:
            - None: skip plotting.
            - A single GaussianBumpAdversary.
            - A list of GaussianBumpAdversary instances (population).
        hps: Hyperparameters namespace (used for dt and n_steps).
        n_replicates: Number of model replicates (logged in title; adversary not vmapped).
    """
    from rlrmp.adversary import GaussianBumpAdversary

    n_timesteps = hps.task.n_steps - 1
    dt = hps.dt
    t = np.arange(n_timesteps) * dt  # (T,) in seconds

    # -----------------------------------------------------------------------
    # Try to load pre-saved profiles from adversary_force_profiles.npz
    # -----------------------------------------------------------------------
    profile_path = results_dir / "adversary_force_profiles.npz"
    pre_saved: dict[str, np.ndarray] | None = None
    if profile_path.exists():
        data = np.load(profile_path)
        pre_saved = {k: data[k] for k in data.files}

    # -----------------------------------------------------------------------
    # Compute profiles from adversary objects if not pre-saved
    # -----------------------------------------------------------------------
    # Represent as a list of (T, 2) arrays (one per population member)
    computed_profiles: list[np.ndarray] | None = None

    if adversaries is not None and pre_saved is None:
        if isinstance(adversaries, GaussianBumpAdversary):
            forces = np.array(adversaries())  # (T, 2)
            computed_profiles = [forces]
        elif isinstance(adversaries, list):
            computed_profiles = [np.array(adv()) for adv in adversaries]  # list of (T,2)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if pre_saved is None and computed_profiles is None:
        print("  No adversary data available for plotting — skipping.")
        return

    n_cols = 1 if pre_saved is not None else 1
    fig, ax = plt.subplots(figsize=(10, 4))

    if pre_saved is not None:
        # Plot one line per SISU key
        sisu_keys = sorted(pre_saved.keys())
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(sisu_keys)))
        for color, key in zip(colors, sisu_keys):
            forces = pre_saved[key]  # (T, 2)
            magnitudes = np.linalg.norm(forces, axis=-1)  # (T,)
            label = key.replace("_", "=").replace("sisu", "SISU ")
            ax.plot(t, magnitudes, color=color, lw=1.5, label=label)
        ax.set_title(
            f"Adversary force magnitude by SISU level (from adversary_force_profiles.npz)\n"
            f"n_replicates={n_replicates}"
        )
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    else:
        # computed_profiles: list of (T, 2)
        magnitudes = np.stack(
            [np.linalg.norm(f, axis=-1) for f in computed_profiles], axis=0
        )  # (K, T)

        K = magnitudes.shape[0]
        if K == 1:
            ax.plot(t, magnitudes[0], color="steelblue", lw=2, label="Adversary force")
            ax.set_title(
                f"Adversary force magnitude (single adversary)\nn_replicates={n_replicates}"
            )
        else:
            cmap = plt.cm.tab10
            for i in range(K):
                ax.plot(
                    t, magnitudes[i], color=cmap(i % 10), lw=1, alpha=0.6, label=f"Adversary {i}"
                )
            mean_mag = magnitudes.mean(axis=0)
            std_mag = magnitudes.std(axis=0)
            ax.plot(t, mean_mag, color="black", lw=2, label="Mean")
            ax.fill_between(
                t,
                mean_mag - std_mag,
                mean_mag + std_mag,
                color="black",
                alpha=0.2,
                label="Mean ± std",
            )
            ax.set_title(
                f"Adversary force magnitude (K={K} population adversaries)\n"
                f"n_replicates={n_replicates}"
            )
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force magnitude (L2 norm)")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    out_path = results_dir / "adversary_force_profiles.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved adversary force profile plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostics for minimax training experiments."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/minimax_test",
        help="Path to a tier results directory (e.g. results/tier1b/mult_single).",
    )
    parser.add_argument(
        "--pert-scale",
        type=float,
        default=5.0,
        help="Perturbation scale for robustness evaluation (default: 5.0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the adversary force profile plot.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Results directory: {results_dir}\n")

    # -----------------------------------------------------------------------
    # Load config and build hyperparameters
    # -----------------------------------------------------------------------
    print("Loading config ... ", end="", flush=True)
    config = load_config(results_dir)
    print("OK")

    hps = build_hps(
        argparse.Namespace(**{k: v for k, v in config.items() if k not in ("git", "output_dir")})
    )

    print("Setting up task-model pair ... ", end="", flush=True)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)
    task = pair.task
    print("OK\n")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print("Loading models:")
    warmup_model = load_model(results_dir, "warmup_model.eqx", hps, config)
    adv_model = load_model(results_dir, "adversarial_model.eqx", hps, config)

    if warmup_model is not None:
        print("  warmup_model.eqx      ... OK")
    else:
        print("  warmup_model.eqx      ... NOT FOUND")

    if adv_model is not None:
        print("  adversarial_model.eqx ... OK")
    else:
        print("  adversarial_model.eqx ... NOT FOUND (training may not be complete)")

    if warmup_model is None and adv_model is None:
        print("\nNo models found — nothing to evaluate.")
        sys.exit(1)

    models_dict: dict = {}
    if warmup_model is not None:
        models_dict["warmup"] = warmup_model
    if adv_model is not None:
        models_dict["adversarial"] = adv_model

    print()

    # -----------------------------------------------------------------------
    # Load adversary
    # -----------------------------------------------------------------------
    print("Loading adversary ... ", end="", flush=True)
    adversary = load_adversary(results_dir, hps)
    if adversary is not None:
        print("OK")
    else:
        print("NOT FOUND (will attempt pre-saved profiles)")
    print()

    # -----------------------------------------------------------------------
    # SECTION 1: SISU weight diagnostics
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("SECTION 1: SISU weight / sensitivity diagnostics")
    print("=" * 100)
    print()

    for name, model in models_dict.items():
        print(f"  [{name}]")
        diag = check_sisu_weights(model, config)
        print(diag["summary"])
        print()

    # -----------------------------------------------------------------------
    # SECTION 2: Comprehensive evaluation metrics
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("SECTION 2: Comprehensive evaluation metrics")
    print("=" * 100)
    print()

    eval_key = jr.PRNGKey(99)
    report_eval_metrics(task, models_dict, pert_scale=args.pert_scale, key=eval_key)

    # -----------------------------------------------------------------------
    # SECTION 3: Adversary force profile plot
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("SECTION 3: Adversary force profile plot")
    print("=" * 100)
    print()

    if args.no_plot:
        print("  Skipped (--no-plot).")
    else:
        plot_adversary_force_profiles(
            results_dir,
            adversaries=adversary,
            hps=hps,
            n_replicates=N_REPLICATES,
        )

    print()
    print("Diagnostics complete.")


if __name__ == "__main__":
    main()
