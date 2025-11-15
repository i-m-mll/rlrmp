"""Visualization tools for loss function structure and temporal patterns.

This module provides utilities to visualize how loss function terms behave across time and trials,
particularly showing time masks, discounts, and their combined effects.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import plotly.graph_objects as go
from feedbax.loss import CompositeLoss, FuncTermsLoss, TargetSpec, TargetStateLoss
from feedbax.task import TaskTrialSpec
from jaxtyping import Array
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def _symlog_transform(data: Array, linear_threshold: float = 1.0) -> Array:
    """Apply symmetric log transformation to handle positive, negative, and zero values.

    The transformation is:
        sign(x) * log10(1 + abs(x) / linear_threshold)

    This allows:
    - Zero maps to zero
    - Small values near zero remain linear
    - Large values are compressed logarithmically
    - Negative values are handled symmetrically

    Arguments:
        data: Input array
        linear_threshold: Values below this are approximately linear

    Returns:
        Transformed array
    """
    return jnp.sign(data) * jnp.log10(1.0 + jnp.abs(data) / linear_threshold)


def _evaluate_temporal_pattern(
    pattern: Optional[Array | Callable],
    trial_specs: TaskTrialSpec,
    n_timesteps: int,
) -> Optional[Array]:
    """Evaluate a time_mask or discount pattern across trials.

    Arguments:
        pattern: Either an array or a callable that takes trial_spec and returns an array
        trial_specs: Batch of trial specifications (batched along first dimension)
        n_timesteps: Number of timesteps in the trial

    Returns:
        Array of shape (n_trials, n_timesteps) or None if pattern is None
    """
    if pattern is None:
        return None

    if callable(pattern):
        # Get number of trials from epoch_bounds (which should be batched)
        epoch_bounds = trial_specs.timeline.epoch_bounds
        n_trials = epoch_bounds.shape[0]

        # Evaluate pattern for each trial by slicing the batched arrays
        # We need to extract single-trial specs from the batched specs
        patterns = []
        for i in range(n_trials):
            # Create a single-trial spec by slicing batched arrays
            single_trial_spec = jax.tree.map(
                lambda x: x[i] if isinstance(x, jnp.ndarray) and x.ndim > 0 else x,
                trial_specs
            )
            pattern_i = pattern(single_trial_spec)
            patterns.append(pattern_i)

        return jnp.stack(patterns, axis=0)
    else:
        # pattern is already an array
        if pattern.ndim == 1:
            # Single pattern for all trials, broadcast
            n_trials = trial_specs.timeline.epoch_bounds.shape[0]
            return jnp.broadcast_to(pattern[None, :], (n_trials, n_timesteps))
        else:
            # Already batched
            return pattern


def _get_epoch_boundaries(trial_specs: TaskTrialSpec) -> Optional[Array]:
    """Extract epoch boundaries from trial specs.

    Returns:
        Array of shape (n_trials, n_epochs+1) or None if not available
    """
    if trial_specs.timeline is None or trial_specs.timeline.epoch_bounds is None:
        return None

    epoch_bounds = trial_specs.timeline.epoch_bounds
    # Handle both batched (n_trials, n_epochs+1) and single (n_epochs+1,)
    if epoch_bounds.ndim == 1:
        # Single trial, add batch dimension
        return epoch_bounds[None, :]
    return epoch_bounds


def _extract_target_state_loss_info(
    term: TargetStateLoss,
    trial_specs: TaskTrialSpec,
    n_timesteps: int,
) -> dict[str, Any]:
    """Extract visualization info from a TargetStateLoss term.

    Arguments:
        term: The TargetStateLoss to analyze
        trial_specs: Batch of trial specifications
        n_timesteps: Number of timesteps in trials

    Returns:
        Dictionary containing:
            - time_mask: (n_trials, n_timesteps) array or None
            - discount: (n_trials, n_timesteps) array or None
            - combined: (n_trials, n_timesteps) array (product of mask and discount)
            - label: term label
    """
    spec = term.spec
    if spec is None:
        # No spec means always active with no discount
        n_trials = jax.tree.leaves(trial_specs)[0].shape[0]
        ones = jnp.ones((n_trials, n_timesteps))
        return {
            "time_mask": ones,
            "discount": ones,
            "combined": ones,
            "label": term.label,
        }

    # Evaluate time_mask and discount
    time_mask = _evaluate_temporal_pattern(spec.time_mask, trial_specs, n_timesteps)
    discount = _evaluate_temporal_pattern(spec.discount, trial_specs, n_timesteps)

    # Compute combined effect
    n_trials = jax.tree.leaves(trial_specs)[0].shape[0]

    # Default to ones if not specified
    if time_mask is None:
        time_mask = jnp.ones((n_trials, n_timesteps))
    if discount is None:
        discount = jnp.ones((n_trials, n_timesteps))

    combined = time_mask * discount

    return {
        "time_mask": time_mask,
        "discount": discount,
        "combined": combined,
        "label": term.label,
    }


def _create_term_heatmap(
    data: Array,
    term_label: str,
    weight: float,
    epoch_bounds: Optional[Array] = None,
    colorscale: str = "Viridis",
    title_suffix: str = "",
) -> go.Heatmap:
    """Create a heatmap showing temporal pattern across trials.

    Arguments:
        data: Array of shape (n_trials, n_timesteps)
        term_label: Label for the loss term
        weight: Weight of this term in the loss
        epoch_bounds: Optional array of shape (n_trials, n_epochs+1)
        colorscale: Plotly colorscale name
        title_suffix: Additional text for title

    Returns:
        Plotly Heatmap trace
    """
    n_trials, n_timesteps = data.shape

    # Create heatmap
    heatmap = go.Heatmap(
        z=data,
        x=list(range(n_timesteps)),
        y=list(range(n_trials)),
        colorscale=colorscale,
        colorbar=dict(title="Value"),
        hovertemplate="Trial: %{y}<br>Time: %{x}<br>Value: %{z:.3f}<extra></extra>",
    )

    return heatmap


def _add_epoch_lines(
    fig: go.Figure,
    epoch_bounds: Array,
    row: int,
    col: int,
    n_trials: int,
) -> None:
    """Add vertical lines showing epoch boundaries.

    Arguments:
        fig: Plotly figure to add lines to
        epoch_bounds: Array of shape (n_trials, n_epochs+1)
        row: Subplot row
        col: Subplot column
        n_trials: Number of trials
    """
    # For visualization, show epoch boundaries for a few example trials
    # or show mean boundaries if they vary
    n_epochs = epoch_bounds.shape[1] - 1

    # Check if boundaries are the same across trials
    if jnp.allclose(epoch_bounds, epoch_bounds[0:1, :]):
        # Same boundaries for all trials, draw once
        for i in range(1, n_epochs + 1):
            fig.add_vline(
                x=float(epoch_bounds[0, i]),
                line_dash="dash",
                line_color="white",
                opacity=0.5,
                row=row,
                col=col,
            )
    # Note: For varying boundaries, we just skip the annotation - it clutters the plot


def visualize_loss_structure(
    loss_fn: CompositeLoss,
    trial_specs: TaskTrialSpec,
    structure_only: bool = True,
    depth: int = 0,
    n_trials_viz: Optional[int] = None,
    show_time_mask: bool = True,
    show_discount: bool = True,
    show_combined: bool = True,
    multiply_term_weight: bool = False,
    log_scale: bool = True,
) -> go.Figure:
    """Visualize the structure and temporal patterns of a loss function.

    Arguments:
        loss_fn: The CompositeLoss to visualize
        trial_specs: Batch of trial specifications to evaluate temporal patterns
        structure_only: If True, only show structure for FuncTermsLoss (don't require states)
        depth: Depth of nesting to visualize (0 = top level only)
        n_trials_viz: Number of trials to show (default: min(10, total))
        show_time_mask: Whether to show time_mask heatmaps
        show_discount: Whether to show discount heatmaps
        show_combined: Whether to show combined (mask * discount) heatmaps
        multiply_term_weight: If True, multiply displayed values by the term's weight in the loss
        log_scale: If True, use symmetric log scale for color mapping (handles zeros naturally)

    Returns:
        Plotly Figure with heatmaps for each loss term
    """
    # Get number of timesteps from trial specs
    if trial_specs.timeline is None or trial_specs.timeline.n_steps is None:
        raise ValueError("trial_specs must have timeline.n_steps")

    n_timesteps = trial_specs.timeline.n_steps - 1  # States are T+1, we skip first
    n_trials_total = jax.tree.leaves(trial_specs)[0].shape[0]

    if n_trials_viz is None:
        n_trials_viz = min(10, n_trials_total)

    # Subset trial specs if needed
    if n_trials_viz < n_trials_total:
        # Slice all arrays in trial_specs
        trial_specs = jax.tree.map(lambda x: x[:n_trials_viz] if isinstance(x, jnp.ndarray) else x, trial_specs)

    # Extract epoch boundaries
    epoch_bounds = _get_epoch_boundaries(trial_specs)

    # Collect info for all terms at the specified depth
    term_infos = []

    if depth == 0:
        # Top level only
        for term_name, term in loss_fn.terms.items():
            weight = loss_fn.weights[term_name]

            if isinstance(term, TargetStateLoss):
                info = _extract_target_state_loss_info(term, trial_specs, n_timesteps)
                info["weight"] = weight
                info["type"] = "TargetStateLoss"

                # Optionally multiply by term weight
                if multiply_term_weight:
                    weight_val = float(weight)
                    info["time_mask"] = info["time_mask"] * weight_val
                    info["discount"] = info["discount"] * weight_val
                    info["combined"] = info["combined"] * weight_val

                term_infos.append(info)
            elif isinstance(term, FuncTermsLoss):
                # For FuncTermsLoss, just show structure
                info = {
                    "label": term.label,
                    "weight": weight,
                    "type": "FuncTermsLoss",
                    "subterms": list(term.terms.keys()),
                    "subweights": term.weights,
                }
                term_infos.append(info)
            elif isinstance(term, CompositeLoss):
                # Treat as black box
                info = {
                    "label": term.label,
                    "weight": weight,
                    "type": "CompositeLoss",
                    "n_subterms": len(term.terms),
                }
                term_infos.append(info)

    # Count how many heatmap columns we need
    n_viz_types = sum([show_time_mask, show_discount, show_combined])
    target_state_terms = [info for info in term_infos if info["type"] == "TargetStateLoss"]
    n_rows = len(term_infos)

    if n_rows == 0:
        logger.warning("No terms to visualize")
        return go.Figure()

    # Create titles for subplots (just the term label)
    titles_for_terms = []
    for info in term_infos:
        if info["type"] == "TargetStateLoss":
            titles_for_terms.append(info['label'])
        else:
            titles_for_terms.append(info['label'])

    # Determine grid layout
    # For mixed types, we need to be clever about layout
    # Simple approach: each row gets n_viz_types columns if TargetStateLoss, 1 if other
    # This is complex, so let's simplify: just show TargetStateLoss terms for now

    if not target_state_terms:
        # No TargetStateLoss terms, just show text summary
        fig = go.Figure()
        text_lines = ["<b>Loss Structure</b><br><br>"]
        for info in term_infos:
            if info["type"] == "FuncTermsLoss":
                subterms_str = ', '.join(f"{k}={v:.2e}" for k, v in info['subweights'].items())
                text_lines.append(
                    f"<b>{info['label']}</b> (w={info['weight']:.2e})<br>"
                    f"Type: FuncTermsLoss<br>"
                    f"Subterms: {subterms_str}<br><br>"
                )
            elif info["type"] == "CompositeLoss":
                text_lines.append(
                    f"<b>{info['label']}</b> (w={info['weight']:.2e})<br>"
                    f"Type: CompositeLoss<br>"
                    f"Subterms: {info['n_subterms']}<br><br>"
                )

        fig.add_annotation(
            text="".join(text_lines),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=12),
        )
        fig.update_layout(
            title="Loss Function Structure (no TargetStateLoss terms to visualize)",
            height=400,
        )
        return fig

    # Create subplots for TargetStateLoss terms only
    n_rows_viz = len(target_state_terms)

    # Minimal vertical spacing between subplots
    vertical_spacing = 0.02

    fig = make_subplots(
        rows=n_rows_viz,
        cols=n_viz_types,
        subplot_titles=titles_for_terms,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.1,
    )

    # Optionally apply symlog transformation
    linear_threshold = 1.0  # Values below this remain approximately linear
    if log_scale:
        for info in target_state_terms:
            if show_time_mask:
                info["time_mask"] = _symlog_transform(info["time_mask"], linear_threshold)
            if show_discount:
                info["discount"] = _symlog_transform(info["discount"], linear_threshold)
            if show_combined:
                info["combined"] = _symlog_transform(info["combined"], linear_threshold)

    # Find global min and max for color scale
    all_values = []
    for info in target_state_terms:
        if show_time_mask:
            all_values.append(jnp.min(info["time_mask"]))
            all_values.append(jnp.max(info["time_mask"]))
        if show_discount:
            all_values.append(jnp.min(info["discount"]))
            all_values.append(jnp.max(info["discount"]))
        if show_combined:
            all_values.append(jnp.min(info["combined"]))
            all_values.append(jnp.max(info["combined"]))

    if all_values:
        global_min = float(jnp.min(jnp.array(all_values)))
        global_max = float(jnp.max(jnp.array(all_values)))
    else:
        global_min = 0.0
        global_max = 1.0

    # Add heatmaps
    for row_idx, info in enumerate(target_state_terms, start=1):
        col_idx = 1
        # Only show colorbar on the last (rightmost) heatmap
        is_last_row = (row_idx == n_rows_viz)

        if show_time_mask:
            heatmap = go.Heatmap(
                z=info["time_mask"],
                x=list(range(n_timesteps)),
                y=list(range(n_trials_viz)),
                colorscale="Greys",
                zmin=global_min,
                zmax=global_max,
                showscale=(col_idx == n_viz_types and is_last_row),
                colorbar=dict(
                    title="symlog" if log_scale else "",
                    len=0.3,
                    y=0.85,
                    yanchor="top"
                ) if (col_idx == n_viz_types and is_last_row) else None,
                hovertemplate="Trial: %{y}<br>Time: %{x}<br>Value: %{z:.3f}<extra></extra>",
            )
            fig.add_trace(heatmap, row=row_idx, col=col_idx)

            # Add epoch lines if available
            if epoch_bounds is not None:
                _add_epoch_lines(fig, epoch_bounds, row_idx, col_idx, n_trials_viz)

            col_idx += 1

        if show_discount:
            heatmap = go.Heatmap(
                z=info["discount"],
                x=list(range(n_timesteps)),
                y=list(range(n_trials_viz)),
                colorscale="Viridis",
                zmin=global_min,
                zmax=global_max,
                showscale=(col_idx == n_viz_types and is_last_row),
                colorbar=dict(
                    title="symlog" if log_scale else "",
                    len=0.3,
                    y=0.85,
                    yanchor="top"
                ) if (col_idx == n_viz_types and is_last_row) else None,
                hovertemplate="Trial: %{y}<br>Time: %{x}<br>Value: %{z:.3f}<extra></extra>",
            )
            fig.add_trace(heatmap, row=row_idx, col=col_idx)

            # Add epoch lines if available
            if epoch_bounds is not None:
                _add_epoch_lines(fig, epoch_bounds, row_idx, col_idx, n_trials_viz)

            col_idx += 1

        if show_combined:
            heatmap = go.Heatmap(
                z=info["combined"],
                x=list(range(n_timesteps)),
                y=list(range(n_trials_viz)),
                colorscale="Plasma",
                zmin=global_min,
                zmax=global_max,
                showscale=(col_idx == n_viz_types and is_last_row),
                colorbar=dict(
                    title="symlog" if log_scale else "",
                    len=0.3,
                    y=0.85,
                    yanchor="top"
                ) if (col_idx == n_viz_types and is_last_row) else None,
                hovertemplate="Trial: %{y}<br>Time: %{x}<br>Value: %{z:.3f}<extra></extra>",
            )
            fig.add_trace(heatmap, row=row_idx, col=col_idx)

            # Add epoch lines if available
            if epoch_bounds is not None:
                _add_epoch_lines(fig, epoch_bounds, row_idx, col_idx, n_trials_viz)

            col_idx += 1

    # Update layout
    fig.update_layout(
        title=f"Loss Function Temporal Patterns ({loss_fn.label})",
        height=max(400, 150 * n_rows_viz),
        showlegend=False,
    )

    # Update axes labels - only show x-axis on bottom row
    for row_idx in range(1, n_rows_viz + 1):
        for col_idx in range(1, n_viz_types + 1):
            # Only show x-axis title and ticks on bottom row
            if row_idx == n_rows_viz:
                fig.update_xaxes(title_text="Time step", row=row_idx, col=col_idx)
            else:
                fig.update_xaxes(title_text="", showticklabels=False, row=row_idx, col=col_idx)

            fig.update_yaxes(title_text="Trial", row=row_idx, col=col_idx)

    return fig
