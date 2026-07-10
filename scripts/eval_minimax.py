"""Evaluate a minimax-trained model and compare warmup vs. adversarially-trained models.

Loads models from a minimax results directory (produced by train_minimax.py) and reports:
  - Velocity at SISU=0 and SISU=1 (pert_scale=0) for warmup and adversarial models
  - Endpoint error and lateral deviation (unperturbed and under perturbation)
  - Adversary's learned force profiles at SISU=0 vs SISU=1

Key comparison: does the adversarially-trained model show a larger SISU velocity effect
than the warmup model?

Expected directory layout (produced by train_minimax.py):
    config.json
    warmup_model.eqx          — pre-adversarial (warm-started) model
    adversarial_model.eqx     — post-adversarial model (present after training completes)
    trained_adversary.eqx     — saved adversary
    adversary_force_profiles.npz  — pre-computed force profiles at SISU=0..1
    warmup_history.eqx        — executor warmup history (warmup phase)
    adversarial_losses.npz    — ctrl_losses and adv_losses (adversarial phase)

Usage:
    uv run python scripts/eval_minimax.py --results-dir results/minimax_test
    uv run python scripts/eval_minimax.py --results-dir results/minimax_test --pert-scale 5.0
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

import jax.random as jr
import numpy as np

from rlrmp.train.adversary import GaussianBumpAdversary
from rlrmp.eval import eval_at_pert0, eval_at_pert_scale
from rlrmp.eval.minimax_io import load_adversary, load_config, load_model
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax_native import build_hps


# ---------------------------------------------------------------------------
# Force profile reporting
# ---------------------------------------------------------------------------


def report_adversary_force_profiles(
    results_dir: Path,
    adversary: GaussianBumpAdversary | None,
    sisu_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    """Print adversary force profile statistics at each SISU value.

    Tries to load pre-computed profiles from adversary_force_profiles.npz first;
    falls back to computing directly from the adversary object.

    Args:
        results_dir: Directory containing adversary_force_profiles.npz.
        adversary: Loaded GaussianBumpAdversary (used as fallback).
        sisu_values: SISU scalars at which to evaluate.
    """
    profile_path = results_dir / "adversary_force_profiles.npz"

    profiles: dict[str, np.ndarray] = {}
    if profile_path.exists():
        data = np.load(profile_path)
        for sisu in sisu_values:
            key = f"sisu_{sisu:.2f}"
            if key in data:
                profiles[key] = data[key]
        print("  (loaded from adversary_force_profiles.npz)")
    elif adversary is not None:
        print("  (computed from loaded adversary)")
        for sisu in sisu_values:
            forces = adversary(float(sisu))  # (T, 2)
            profiles[f"sisu_{sisu:.2f}"] = np.array(forces)
    else:
        print("  No adversary or pre-saved profiles found.")
        return

    hdr = f"  {'SISU':>6} | {'L2 norm':>10} | {'peak_x':>10} | {'peak_y':>10} | {'rms':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for sisu in sisu_values:
        key = f"sisu_{sisu:.2f}"
        if key not in profiles:
            print(f"  {sisu:>6.2f} | MISSING")
            continue
        f = profiles[key]  # (T, 2)
        l2 = float(np.linalg.norm(f))
        peak_x = float(np.max(np.abs(f[:, 0])))
        peak_y = float(np.max(np.abs(f[:, 1])))
        rms = float(np.sqrt(np.mean(f ** 2)))
        print(f"  {sisu:>6.2f} | {l2:>10.4f} | {peak_x:>10.4f} | {peak_y:>10.4f} | {rms:>10.4f}")

    # Highlight SISU=0 vs SISU=1 ratio (core diagnostic)
    k0, k1 = "sisu_0.00", "sisu_1.00"
    if k0 in profiles and k1 in profiles:
        norm0 = float(np.linalg.norm(profiles[k0]))
        norm1 = float(np.linalg.norm(profiles[k1]))
        ratio = norm1 / (norm0 + 1e-8)
        print(f"\n  Force norm ratio SISU=1 / SISU=0: {ratio:.3f}")
        if ratio > 1.2:
            print("  → Adversary has learned to apply stronger forces when SISU=1 (GOOD)")
        elif ratio < 0.8:
            print("  → Adversary applies weaker forces when SISU=1 (unexpected)")
        else:
            print("  → Forces are approximately uniform across SISU (adversary not SISU-conditional)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a minimax-trained model and its adversary."
    )
    parser.add_argument(
        "--results-dir", type=str, default="_artifacts/minimax/minimax_test",
        help=(
            "Path to minimax bulk artifact directory containing the trained model. "
            "Default: _artifacts/minimax/minimax_test. "
            "Use rlrmp.paths.run_artifact_dir(exp, run) to construct this path."
        ),
    )
    parser.add_argument(
        "--pert-scale", type=float, default=5.0,
        help="Perturbation scale for robustness evaluation (default: 5.0).",
    )
    parser.add_argument(
        "--sisu-pert", type=float, default=0.5,
        help="SISU value for perturbation robustness evaluation (default: 0.5).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Results directory: {results_dir}\n")

    # -----------------------------------------------------------------------
    # Load config and set up task
    # -----------------------------------------------------------------------
    print("Loading config ... ", end="", flush=True)
    config = load_config(results_dir)
    print("OK")

    hps = build_hps(argparse.Namespace(**{
        k: v for k, v in config.items()
        if k not in ("git", "output_dir")
    }))

    print("Setting up task-model pair ... ", end="", flush=True)
    key = jr.PRNGKey(42)
    pair = setup_task_model_pair(hps, key=key)
    task = pair.task
    print("OK\n")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print("Loading models:")
    # load_model returns already-squeezed models (handles both unsqueezed and squeezed on-disk)
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

    print()

    # -----------------------------------------------------------------------
    # Load adversary
    # -----------------------------------------------------------------------
    print("Loading adversary ... ", end="", flush=True)
    adversary = load_adversary(results_dir, hps)
    if adversary is not None:
        print("OK")
    else:
        print("NOT FOUND")
    print()

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    SISU_VALUES = [0.0, 0.5, 1.0]
    pert_scale = args.pert_scale
    sisu_pert = args.sisu_pert

    models_to_eval = []
    if warmup_model is not None:
        models_to_eval.append(("warmup", warmup_model))
    if adv_model is not None:
        models_to_eval.append(("adversarial", adv_model))

    # Table 1: SISU velocity effect (pert_scale=0)
    print("=" * 90)
    print("TABLE 1: SISU velocity effect (pert_scale=0)")
    print("=" * 90)
    print()

    t1_results = {}
    for name, model in models_to_eval:
        row = {"name": name}
        eval_key = jr.PRNGKey(10)
        for sisu in SISU_VALUES:
            eval_key, k = jr.split(eval_key)
            km = eval_at_pert0(task, model, sisu=sisu, key=k)
            row[f"vel_{sisu}"] = float(np.mean(km["peak_velocity"]))
            row[f"ep_err_{sisu}"] = float(np.mean(km["endpoint_error"]))
            row[f"lat_dev_{sisu}"] = float(np.mean(km["max_lateral_deviation"]))
        t1_results[name] = row

    hdr1 = (
        f"  {'model':>12} | {'vel(S=0)':>9} | {'vel(S=0.5)':>10} | {'vel(S=1)':>9} | "
        f"{'Δvel(0→1)':>10} | {'Δvel%':>8} | {'ep_err(S=0.5)':>14}"
    )
    print(hdr1)
    print("  " + "-" * (len(hdr1) - 2))
    for name, row in t1_results.items():
        delta = row["vel_1.0"] - row["vel_0.0"]
        delta_pct = 100.0 * delta / (row["vel_0.0"] + 1e-8)
        print(
            f"  {name:>12} | {row['vel_0.0']:>9.4f} | {row['vel_0.5']:>10.4f} | "
            f"{row['vel_1.0']:>9.4f} | {delta:>+10.4f} | {delta_pct:>+7.2f}% | "
            f"{row['ep_err_0.5']:>14.4f}"
        )
    print()

    # Compare warmup vs adversarial if both are available
    if "warmup" in t1_results and "adversarial" in t1_results:
        w = t1_results["warmup"]
        a = t1_results["adversarial"]
        delta_w = w["vel_1.0"] - w["vel_0.0"]
        delta_a = a["vel_1.0"] - a["vel_0.0"]
        print(f"  SISU velocity effect (Δvel = vel(1) - vel(0)):")
        print(f"    warmup:       {delta_w:+.4f}")
        print(f"    adversarial:  {delta_a:+.4f}")
        if delta_a > delta_w + 0.005:
            print("  → Adversarial training INCREASED the SISU velocity effect")
        elif delta_a < delta_w - 0.005:
            print("  → Adversarial training DECREASED the SISU velocity effect")
        else:
            print("  → No meaningful change in SISU velocity effect")
        print()

    # Table 2: Robustness under perturbation
    print("=" * 90)
    print(f"TABLE 2: Robustness under perturbation (pert_scale={pert_scale}, SISU={sisu_pert})")
    print("=" * 90)
    print()

    hdr2 = (
        f"  {'model':>12} | {'vel(p)':>9} | {'ep_err(p)':>10} | {'lat_dev(p)':>12}"
    )
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for name, model in models_to_eval:
        eval_key = jr.PRNGKey(50)
        eval_key, k = jr.split(eval_key)
        km = eval_at_pert_scale(task, model, sisu=sisu_pert, pert_scale=pert_scale, key=k)
        vel = float(np.mean(km["peak_velocity"]))
        ep_err = float(np.mean(km["endpoint_error"]))
        lat_dev = float(np.mean(km["max_lateral_deviation"]))
        print(
            f"  {name:>12} | {vel:>9.4f} | {ep_err:>10.4f} | {lat_dev:>12.4f}"
        )
    print()

    # Table 3: Adversary force profiles
    print("=" * 90)
    print("TABLE 3: Adversary force profiles (SISU=0 vs SISU=1)")
    print("=" * 90)
    print()
    report_adversary_force_profiles(results_dir, adversary)
    print()

    # Adversarial loss curves (if available)
    adv_losses_path = results_dir / "adversarial_losses.npz"
    if adv_losses_path.exists():
        data = np.load(adv_losses_path)
        ctrl_losses = data["ctrl_losses"]
        adv_losses = data["adv_losses"]
        n_batches = len(ctrl_losses)
        print("=" * 90)
        print(f"Adversarial training loss summary ({n_batches} batches):")
        print(f"  controller loss: initial={ctrl_losses[0]:.4f}, final={ctrl_losses[-1]:.4f}")
        print(f"  adversary loss:  initial={adv_losses[0]:.4f}, final={adv_losses[-1]:.4f}")
        print()

    print("Notes:")
    print(f"  - Table 1: pert_scale=0 (perturbations zeroed), varying SISU only.")
    print(f"  - Table 2: pert_scale={pert_scale}, SISU={sisu_pert}.")
    print(f"  - Table 3: adversary force profiles — key metric is SISU=1/SISU=0 norm ratio.")
    print(f"  - warmup = pre-adversarial model (warm-start only).")
    print(f"  - adversarial = post-adversarial model (after minimax training).")


if __name__ == "__main__":
    main()
