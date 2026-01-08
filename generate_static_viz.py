"""Generate static PNG version of loss visualization."""

import jax
import jax.numpy as jnp
import yaml
from feedbax.task import DelayedReaches
from feedbax.types import TreeNamespace

from rlrmp.loss import get_reach_loss
from rlrmp.viz import visualize_loss_structure


def dict_to_namespace(d):
    """Recursively convert dict to TreeNamespace."""
    if isinstance(d, dict):
        # Only process string keys
        converted = {}
        for k, v in d.items():
            if isinstance(k, str):
                converted[k] = dict_to_namespace(v)
            else:
                # Skip non-string keys
                converted[str(k)] = dict_to_namespace(v)
        return TreeNamespace(**converted)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


def main():
    # Load the default config directly from YAML
    config_path = "src/rlrmp/config/modules/training/default.yml"
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    hps = dict_to_namespace(cfg_dict)

    print("Building loss function...")
    loss_fn = get_reach_loss(hps)

    # Build task kwargs (including loss_func which is required)
    task_cfg = hps.task
    task_kwargs = {
        "loss_func": loss_fn,
        "n_steps": task_cfg.n_steps,
        "workspace": jnp.array(task_cfg.workspace),
    }

    # Add delayed reach specific parameters
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

    # Generate validation trial specs
    key = jax.random.PRNGKey(0)
    trial_specs = task.get_validation_trials(key)

    print(f"Generating visualization for {trial_specs.timeline.epoch_bounds.shape[0]} trials...")

    # Create visualization - only show combined for cleaner view
    # Uses defaults: multiply_term_weight=False, log_scale=True (symlog)
    fig = visualize_loss_structure(
        loss_fn,
        trial_specs,
        structure_only=True,
        depth=0,
        n_trials_viz=7,
        show_time_mask=False,
        show_discount=False,
        show_combined=True,
    )

    # Save to HTML
    output_html = "loss_structure_combined.html"
    fig.write_html(output_html)
    print(f"HTML saved to: {output_html}")

    # Try to export to PNG if kaleido is available
    try:
        output_png = "loss_structure_combined.png"
        fig.write_image(output_png, width=1600, height=2000)
        print(f"PNG saved to: {output_png}")
    except Exception as e:
        print(f"Could not save PNG (kaleido might not be installed): {e}")
        print("To enable PNG export, install: pip install kaleido")


if __name__ == "__main__":
    main()
