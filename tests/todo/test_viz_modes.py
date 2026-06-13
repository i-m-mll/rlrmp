"""Test script to demonstrate different visualization modes."""

import jax
import jax.numpy as jnp
import yaml
from feedbax import DelayedReaches
from feedbax_experiments.types import TreeNamespace

from rlrmp.loss import get_reach_loss
from rlrmp.viz import visualize_loss_structure


def dict_to_namespace(d):
    """Recursively convert dict to TreeNamespace."""
    if isinstance(d, dict):
        converted = {}
        for k, v in d.items():
            if isinstance(k, str):
                converted[k] = dict_to_namespace(v)
            else:
                converted[str(k)] = dict_to_namespace(v)
        return TreeNamespace(**converted)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def setup():
    """Load config and generate trial specs."""
    config_path = "src/rlrmp/config/modules/training/default.yml"
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    hps = dict_to_namespace(cfg_dict)

    loss_fn = get_reach_loss(hps)

    task_cfg = hps.task
    task_kwargs = {
        "loss_func": loss_fn,
        "n_steps": task_cfg.n_steps,
        "workspace": jnp.array(task_cfg.workspace),
    }

    if hasattr(task_cfg, "epoch_len_ranges"):
        task_kwargs["epoch_len_ranges"] = task_cfg.epoch_len_ranges
    if hasattr(task_cfg, "target_on_epochs"):
        task_kwargs["target_on_epochs"] = task_cfg.target_on_epochs
    if hasattr(task_cfg, "hold_epochs"):
        task_kwargs["hold_epochs"] = task_cfg.hold_epochs
    if hasattr(task_cfg, "move_epochs"):
        task_kwargs["move_epochs"] = task_cfg.move_epochs
    if hasattr(task_cfg, "p_catch_trial"):
        task_kwargs["p_catch_trial"] = task_cfg.p_catch_trial

    task = DelayedReaches(**task_kwargs)
    key = jax.random.PRNGKey(0)
    trial_specs = task.get_validation_trials(key)

    return loss_fn, trial_specs


def main():
    print("Setting up...")
    loss_fn, trial_specs = setup()
    n_trials = trial_specs.timeline.epoch_bounds.shape[0]
    print(f"Generated {n_trials} trial specs")

    # Mode 1: Temporal patterns only (no weights)
    print("\n1. Creating temporal patterns visualization (no weights)...")
    fig1 = visualize_loss_structure(
        loss_fn,
        trial_specs,
        structure_only=True,
        n_trials_viz=7,
        show_time_mask=False,
        show_discount=False,
        show_combined=True,
        multiply_term_weight=False,
        log_scale=False,
    )
    fig1.write_html("loss_viz_temporal_patterns.html")
    print("   Saved: loss_viz_temporal_patterns.html")

    # Mode 2: With term weights (linear scale)
    print("\n2. Creating weighted contributions visualization (linear scale)...")
    fig2 = visualize_loss_structure(
        loss_fn,
        trial_specs,
        structure_only=True,
        n_trials_viz=7,
        show_time_mask=False,
        show_discount=False,
        show_combined=True,
        multiply_term_weight=True,
        log_scale=False,
    )
    fig2.write_html("loss_viz_weighted_linear.html")
    print("   Saved: loss_viz_weighted_linear.html")

    # Mode 3: With term weights (log scale)
    print("\n3. Creating weighted contributions visualization (symlog scale)...")
    fig3 = visualize_loss_structure(
        loss_fn,
        trial_specs,
        structure_only=True,
        n_trials_viz=7,
        show_time_mask=False,
        show_discount=False,
        show_combined=True,
        multiply_term_weight=True,
        log_scale=True,
    )
    fig3.write_html("loss_viz_weighted_symlog.html")
    print("   Saved: loss_viz_weighted_symlog.html")

    # Also save PNGs if possible
    try:
        print("\n4. Exporting PNG versions...")
        fig1.write_image("loss_viz_temporal_patterns.png", width=1600, height=1400)
        fig2.write_image("loss_viz_weighted_linear.png", width=1600, height=1400)
        fig3.write_image("loss_viz_weighted_symlog.png", width=1600, height=1400)
        print("   PNGs saved successfully!")
    except Exception as e:
        print(f"   Could not save PNGs: {e}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("Mode 1 (temporal patterns): Shows relative temporal shapes")
    print("  - Range: 0 to ~3 (discount factors)")
    print("  - Use: Compare how different terms vary over time")
    print()
    print("Mode 2 (weighted, linear): Shows actual loss contributions")
    print("  - Range: 0 to ~10 (hold terms dominate)")
    print("  - Use: See which terms dominate when similar magnitude")
    print()
    print("Mode 3 (weighted, symlog): Shows all terms clearly")
    print("  - Range: symlog scale handles 1e-6 to 10")
    print("  - Use: See all terms when some dominate by orders of magnitude")
    print("="*60)


if __name__ == "__main__":
    main()
