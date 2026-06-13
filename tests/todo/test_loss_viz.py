"""Test script for loss function visualization."""

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
    print(f"Loss function: {loss_fn.label}")
    print(f"Terms: {list(loss_fn.terms.keys())}")
    print(f"Weights: {loss_fn.weights}")

    # Create a task to generate trial specs
    print("\nGenerating trial specs...")
    task_cfg = hps.task

    # Build task kwargs (including loss_func which is required)
    task_kwargs = {
        "loss_func": loss_fn,  # Required by DelayedReaches
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

    n_trials = trial_specs.timeline.epoch_bounds.shape[0]
    print(f"Generated {n_trials} trial specs")
    print(f"Timeline n_steps: {trial_specs.timeline.n_steps}")
    print(f"Epoch bounds shape: {trial_specs.timeline.epoch_bounds.shape}")
    print(f"First trial epoch bounds: {trial_specs.timeline.epoch_bounds[0]}")

    # Visualize
    print("\nCreating visualization...")
    fig = visualize_loss_structure(
        loss_fn,
        trial_specs,
        structure_only=True,
        depth=0,
        n_trials_viz=10,
        show_time_mask=True,
        show_discount=True,
        show_combined=True,
    )

    # Save to HTML
    output_path = "loss_structure_viz.html"
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")
    print("Open this file in a web browser to view the interactive plot.")


if __name__ == "__main__":
    main()
