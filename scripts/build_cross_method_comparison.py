"""Build cross_method_comparison.md from flavor-B summary + first-run baselines.

Bug: 74bfd86 — companion to ``run_induced_gain_flavor_b.py``.

Reads:
- ``_artifacts/part2_5/runs/induced_gain_flavor_b/summary.json`` (this run)
- First-run baselines hard-coded from
  ``results/part2_5/runs/induced_gain_first_run/notes.md`` (since the corresponding
  ``gains.npz`` is on a separate branch and not present in this worktree).

Writes:
- ``results/part2_5/induced_gain_flavor_b/cross_method_comparison.md``

Usage:
    uv run python scripts/build_cross_method_comparison.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Hard-coded from results/part2_5/runs/induced_gain_first_run/notes.md.
# Single replicate (replicate 0) per group. The mult_single rep-0 row is the
# documented degenerate replicate; the corrected rep-2 value is also recorded
# for reference.
FIRST_RUN_BASELINES: list[dict] = [
    {"group": "baseline_standard_12k", "g_af": 0.1237, "g_sd": 148.5024, "g_sp": 2.6132,
     "category": "flavor-A baseline (warmup only)"},
    {"group": "vanilla_single",       "g_af": 0.2481, "g_sd": 169.2627, "g_sp": 4.3254,
     "category": "flavor-A baseline (no adversary)"},
    {"group": "vanilla_pop5",         "g_af": 0.1458, "g_sd": 164.6446, "g_sp": 4.4057,
     "category": "flavor-A baseline (no adversary)"},
    {"group": "minimax_single_seed0", "g_af": 0.1446, "g_sd": 153.3508, "g_sp": 2.1396,
     "category": "flavor-A APT minimax"},
    {"group": "minimax_single_seed1", "g_af": 0.1563, "g_sd": 162.8198, "g_sp": 1.3214,
     "category": "flavor-A APT minimax"},
    {"group": "minimax_single_seed2", "g_af": 0.1556, "g_sd": 162.9178, "g_sp": 1.4541,
     "category": "flavor-A APT minimax"},
    {"group": "mult_single (rep0; degen.)", "g_af": 15.6446, "g_sd": 5863.8732, "g_sp": 605.3933,
     "category": "flagged degenerate"},
    {"group": "mult_single (rep2; replacement)", "g_af": 0.1834, "g_sd": float("nan"), "g_sp": float("nan"),
     "category": "flavor-A multiplicative (corrected)"},
    {"group": "mult_pop5",            "g_af": 0.1789, "g_sd": 165.0563, "g_sp": 4.2192,
     "category": "flavor-A multiplicative"},
    {"group": "ratio03_single",       "g_af": 0.1709, "g_sd": 163.3942, "g_sp": 2.6417,
     "category": "flavor-A ratio03"},
    {"group": "ratio03_pop5",         "g_af": 0.1585, "g_sd": 164.5323, "g_sp": 2.1005,
     "category": "flavor-A ratio03"},
]


def _aggregate_by_eta(summary_groups: list[dict], ch_label: str) -> dict[float, dict]:
    """Aggregate per-eta median + spread, pooling across the 3 seeds × 5 reps."""
    by_eta: dict[float, list[float]] = {}
    by_eta_outliers: dict[float, int] = {}
    for g in summary_groups:
        eta = float(g["eta_max"])
        ch = g.get(ch_label, {})
        gammas = ch.get("gammas", [])
        flags = ch.get("outlier_flags", [])
        # Pool only non-outlier replicates per the §5.2 hygiene rule.
        kept = [v for v, f in zip(gammas, flags) if (not f) and np.isfinite(v)]
        by_eta.setdefault(eta, []).extend(kept)
        by_eta_outliers[eta] = by_eta_outliers.get(eta, 0) + sum(flags)

    out: dict[float, dict] = {}
    for eta, vals in by_eta.items():
        if not vals:
            out[eta] = {
                "median": float("nan"), "mad": float("nan"), "n_kept": 0,
                "n_outliers": by_eta_outliers.get(eta, 0),
                "min": float("nan"), "max": float("nan"),
            }
            continue
        arr = np.asarray(vals)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        out[eta] = {
            "median": med, "mad": mad, "n_kept": len(arr),
            "n_outliers": by_eta_outliers.get(eta, 0),
            "min": float(arr.min()), "max": float(arr.max()),
        }
    return out


def render(summary_path: Path, out_path: Path):
    summary = json.loads(summary_path.read_text())
    g_star = float(summary.get("gamma_star_riccati", float("nan")))
    groups = summary["groups"]

    sd_label = "structural_da__qr_cost"
    af_label = "additive_force__qr_cost"
    sp_label = "sensory_perturbation__qr_cost"

    sd_by_eta = _aggregate_by_eta(groups, sd_label)
    af_by_eta = _aggregate_by_eta(groups, af_label)
    sp_by_eta = _aggregate_by_eta(groups, sp_label)

    # Flavor-A baselines: cross-method median for the headline channel,
    # excluding the documented degenerate ``mult_single`` rep-0 row.
    flavor_a_sd_vals = [b["g_sd"] for b in FIRST_RUN_BASELINES
                        if np.isfinite(b["g_sd"]) and "degen" not in b["group"]]
    flavor_a_sd_median = float(np.median(flavor_a_sd_vals)) if flavor_a_sd_vals else float("nan")

    # Build the document.
    lines: list[str] = []
    lines.append("# Cross-method induced-gain comparison: Part 2.5 baselines vs flavor-(b)")
    lines.append("")
    lines.append("**Issue.** `74bfd86`")
    lines.append("**Branch.** `feature/induced-gain-flavor-b`")
    lines.append("**Companion run-spec.** [`run.json`](../runs/induced_gain_flavor_b/run.json)")
    lines.append("**Date.** 2026-05-08")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Plant: rlrmp regime, `linearize_pointmass(mass=1.0, damping=10.0, tau=0.05, dt=0.01)`.")
    lines.append(f"- Cost: `cost_schedule_from_spec(CostSpec(n_steps={summary.get('horizon')}))` (`Q_f=1.0`).")
    lines.append("- Reach: `(0,0) -> (0.15, 0.0)` (15 cm forward), held mid-movement (hold=0, go=1).")
    lines.append("- SISU: 0.5.")
    lines.append(f"- Algorithm: power iteration only (3 restarts, max_iter={summary.get('max_iter')}, **rtol={summary.get('rtol')}**, post-probe canonical).")
    lines.append(f"- Riccati baseline: `gamma_star = {g_star:.6f}`.")
    lines.append("- Replicate hygiene rule: flag `gamma > 10x` from group median (or `< median / 10`); flagged replicates excluded from per-eta medians.")
    lines.append("")
    lines.append("Pre-registered headline metric: **`gamma_sd x qr_cost`** (induced gain on the")
    lines.append("`structural_da` w channel against cost-matched z). This is the channel that")
    lines.append("was uniformly large (~150–170) across all flavor-A trained controllers in")
    lines.append("the first run — flavor-B training is the manipulation that should shift it")
    lines.append("if the (a) ⊊ (b) thesis is empirically supported.")
    lines.append("")
    lines.append("## Table 1 — Headline `gamma_sd x qr_cost`")
    lines.append("")
    lines.append("Flavor-A first-run values from `results/part2_5/runs/induced_gain_first_run/notes.md`")
    lines.append("(single replicate per group, replicate 0). Flavor-B values from this run; per-eta")
    lines.append("median + MAD across **non-outlier** replicates (3 seeds × 5 reps − flagged).")
    lines.append("")
    lines.append("| Method / config | `gamma_sd` (median) | spread | `gamma_sd / gstar` | n_kept / n_outliers | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for b in FIRST_RUN_BASELINES:
        if "degen" in b["group"]:
            note = "documented degenerate rep-0; first-run replaced with rep-2"
        else:
            note = b["category"]
        if np.isfinite(b["g_sd"]):
            ratio = b["g_sd"] / g_star
            lines.append(f"| `{b['group']}` | {b['g_sd']:.2f} | (single rep) | {ratio:.0f} | 1/0 | {note} |")
        else:
            lines.append(f"| `{b['group']}` | — | — | — | 1/0 | {note} |")
    for eta in sorted(sd_by_eta.keys()):
        s = sd_by_eta[eta]
        if np.isfinite(s["median"]):
            ratio = s["median"] / g_star
            lines.append(
                f"| **`flavor_b_eta{eta:.2f}`** (3 seeds × 5 reps) | "
                f"**{s['median']:.2f}** | MAD={s['mad']:.2f} (range {s['min']:.2f}–{s['max']:.2f}) | "
                f"{ratio:.0f} | {s['n_kept']}/{s['n_outliers']} | flavor-B (this run) |"
            )
        else:
            lines.append(
                f"| **`flavor_b_eta{eta:.2f}`** | — | — | — | "
                f"0/{s['n_outliers']} | flavor-B (this run) — all replicates flagged |"
            )
    lines.append("")
    lines.append(f"Flavor-A cross-method median (excluding `mult_single` rep-0 degenerate): **{flavor_a_sd_median:.2f}**.")
    lines.append("")
    # Headline ratio
    flavor_b_overall = []
    for s in sd_by_eta.values():
        if np.isfinite(s["median"]):
            flavor_b_overall.append(s["median"])
    flavor_b_overall_median = float(np.median(flavor_b_overall)) if flavor_b_overall else float("nan")
    if np.isfinite(flavor_b_overall_median) and np.isfinite(flavor_a_sd_median) and flavor_a_sd_median > 0:
        ratio = flavor_b_overall_median / flavor_a_sd_median
        lines.append(f"**Headline ratio**: flavor-B `gamma_sd` median = **{flavor_b_overall_median:.2f}**, ")
        lines.append(f"flavor-A `gamma_sd` median = **{flavor_a_sd_median:.2f}**, ratio = **{ratio:.3f}**.")
    lines.append("")

    lines.append("## Table 2 — Auxiliary channels (`gamma_af`, `gamma_sp`)")
    lines.append("")
    lines.append("| Method / config | `gamma_af` | `gamma_sp` |")
    lines.append("|---|---|---|")
    for b in FIRST_RUN_BASELINES:
        if "degen" in b["group"] or "replacement" in b["group"]:
            af_str = f"{b['g_af']:.4f}"
            sp_str = f"{b['g_sp']:.2f}" if np.isfinite(b["g_sp"]) else "—"
        else:
            af_str = f"{b['g_af']:.4f}"
            sp_str = f"{b['g_sp']:.2f}"
        lines.append(f"| `{b['group']}` | {af_str} | {sp_str} |")
    for eta in sorted(af_by_eta.keys()):
        a = af_by_eta[eta]
        s = sp_by_eta[eta]
        af_str = f"{a['median']:.4f} (MAD={a['mad']:.4f}, n={a['n_kept']}/{a['n_outliers']})" if np.isfinite(a["median"]) else "—"
        sp_str = f"{s['median']:.2f} (MAD={s['mad']:.2f}, n={s['n_kept']}/{s['n_outliers']})" if np.isfinite(s["median"]) else "—"
        lines.append(f"| **`flavor_b_eta{eta:.2f}`** | {af_str} | {sp_str} |")
    lines.append("")

    # Per-(eta, seed) detail
    lines.append("## Table 3 — Per-(eta, seed) flavor-B detail")
    lines.append("")
    lines.append("| Group | n_kept / n_outliers | `gamma_sd` median | `gamma_af` median | `gamma_sp` median |")
    lines.append("|---|---|---|---|---|")
    for g in groups:
        sd = g.get(sd_label, {})
        af = g.get(af_label, {})
        sp = g.get(sp_label, {})
        n_kept = sum(1 for f in sd.get("outlier_flags", []) if not f)
        n_out = sd.get("n_excluded_outliers", 0)
        sd_med = sd.get("median", float("nan"))
        af_med = af.get("median", float("nan"))
        sp_med = sp.get("median", float("nan"))
        sd_str = f"{sd_med:.2f}" if np.isfinite(sd_med) else "—"
        af_str = f"{af_med:.4f}" if np.isfinite(af_med) else "—"
        sp_str = f"{sp_med:.2f}" if np.isfinite(sp_med) else "—"
        lines.append(f"| `{g['group']}` | {n_kept}/{n_out} | {sd_str} | {af_str} | {sp_str} |")
    lines.append("")

    # Eta_max trend (flavor-B-only)
    lines.append("## eta_max trend (flavor-B sweep)")
    lines.append("")
    lines.append("Per-eta median across all kept replicates (after hygiene exclusion):")
    lines.append("")
    lines.append("| eta_max | `gamma_sd` median | `gamma_af` median | `gamma_sp` median |")
    lines.append("|---|---|---|---|")
    for eta in sorted(sd_by_eta.keys()):
        sd = sd_by_eta[eta]
        af = af_by_eta[eta]
        sp = sp_by_eta[eta]
        sd_str = f"{sd['median']:.2f}" if np.isfinite(sd["median"]) else "—"
        af_str = f"{af['median']:.4f}" if np.isfinite(af["median"]) else "—"
        sp_str = f"{sp['median']:.2f}" if np.isfinite(sp["median"]) else "—"
        lines.append(f"| {eta:.2f} | {sd_str} | {af_str} | {sp_str} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path", type=Path,
        default=Path("_artifacts/part2_5/runs/induced_gain_flavor_b/summary.json"),
    )
    parser.add_argument(
        "--out-path", type=Path,
        default=Path("results/part2_5/induced_gain_flavor_b/cross_method_comparison.md"),
    )
    args = parser.parse_args()
    render(args.summary_path, args.out_path)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
