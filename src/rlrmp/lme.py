"""
Linear Mixed Effects Models for PyTree-structured data.

This module provides functions to fit linear mixed effects models where the data
is organized in PyTree structures with hierarchical regressors.
"""

import itertools
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from jax_cookbook import LDict, LDictConstructor
from statsmodels.regression.mixed_linear_model import MixedLMResults


def _get_tree_level_info(tree, is_leaf=None):
    """Extract information about tree levels and their values."""
    # Get the types/labels along the path to the first leaf
    level_types = jtree.tree_level_types(tree, is_leaf=is_leaf or (lambda x: False))

    # For each level, collect the unique values
    level_values = []
    current = tree

    for level_type in level_types:
        if isinstance(level_type, LDictConstructor):
            # It's an LDict level
            if isinstance(current, LDict):
                values = list(current.keys())
                level_values.append((level_type.label, values))
                # Move to first child for next iteration
                if values:
                    current = current[values[0]]
            else:
                raise ValueError(f"Expected LDict with label {level_type.label}")
        elif isinstance(level_type, type):
            # It's a regular type level (tuple, list, etc.)
            if hasattr(current, "__iter__") and not isinstance(current, (str, bytes)):
                if isinstance(current, dict):
                    values = list(current.keys())
                else:
                    values = list(range(len(current)))
                level_values.append((level_type.__name__, values))
                # Move to first child
                if values:
                    if isinstance(current, dict):
                        current = current[values[0]]
                    else:
                        current = current[0]
            else:
                break

    return level_types, level_values


def _flatten_pytree_to_dataframe(
    response_tree,
    extra_regressors: Optional[Dict[str, Any]] = None,
    subject_axis: Optional[int] = None,
    normalize: Union[bool, Dict, None] = None,
    regressor_overrides: Optional[Dict] = None,
) -> pd.DataFrame:
    """Convert PyTree structure to a pandas DataFrame suitable for mixed effects modeling."""

    # Get tree structure information
    level_types, level_values = _get_tree_level_info(response_tree)

    # Build a list to collect all data rows
    data_rows = []

    # Helper to get regressor values from path
    def path_to_regressors(path_keys):
        regressors = {}
        for i, (level_name, _) in enumerate(level_values):
            if i < len(path_keys):
                key = path_keys[i]
                # Use override if provided
                if regressor_overrides and level_name in regressor_overrides:
                    override_node = regressor_overrides[level_name]
                    if hasattr(override_node, "__getitem__"):
                        key_val = (
                            override_node[key]
                            if isinstance(override_node, dict)
                            else override_node[key]
                        )
                    else:
                        key_val = float(key)  # Fallback to numeric
                else:
                    # Use key directly (for LDict) or convert to float (for indices)
                    key_val = float(key) if isinstance(key, (int, np.integer)) else key
                regressors[level_name] = key_val
        return regressors

    # Flatten the response tree and collect data
    leaves_with_path = jt.leaves_with_path(response_tree)

    for path, response_array in leaves_with_path:
        # Extract keys from the path
        path_keys = [jtree.node_key_to_value(k) for k in path]

        # Get regressor values for this path
        regressors = path_to_regressors(path_keys)

        # Get extra regressor values if provided
        extra_values = {}
        if extra_regressors:
            for extra_name, extra_tree in extra_regressors.items():
                # Navigate to the same leaf in the extra tree
                extra_leaf = extra_tree
                for key in path:
                    extra_leaf = jtree.get_child_node_given_key(extra_leaf, key)

                # Broadcast if necessary
                if hasattr(extra_leaf, "shape"):
                    extra_broadcasted = jnp.broadcast_to(extra_leaf, response_array.shape)
                    extra_values[extra_name] = extra_broadcasted
                else:
                    extra_values[extra_name] = extra_leaf

        # Flatten the arrays and create rows
        response_flat = np.asarray(response_array).flatten()
        n_samples = len(response_flat)

        # Handle subject axis if specified
        if subject_axis is not None:
            subject_ids = np.arange(response_array.shape[subject_axis])
            # Repeat subject IDs for all other dimensions
            subject_ids_flat = np.repeat(subject_ids, n_samples // len(subject_ids))
        else:
            subject_ids_flat = None

        # Create rows for this leaf
        for i in range(n_samples):
            row = {"response": response_flat[i]}
            row.update(regressors)

            # Add extra regressors
            for extra_name, extra_array in extra_values.items():
                if hasattr(extra_array, "shape"):
                    row[extra_name] = np.asarray(extra_array).flatten()[i]
                else:
                    row[extra_name] = extra_array

            # Add subject ID if applicable
            if subject_ids_flat is not None:
                row["subject"] = subject_ids_flat[i]

            data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Apply normalization
    if normalize is not False and normalize is not None:
        df = _normalize_dataframe(df, normalize)
    elif normalize is None:
        # Default normalization: center and scale all numeric columns
        df = _normalize_dataframe(df, {})

    return df


def _normalize_dataframe(df: pd.DataFrame, normalize_spec: Union[Dict, bool]) -> pd.DataFrame:
    """Normalize columns in the dataframe."""
    df = df.copy()

    if normalize_spec is False:
        return df

    # Default: normalize all numeric columns except 'subject'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "subject" in numeric_cols:
        numeric_cols.remove("subject")

    for col in numeric_cols:
        if isinstance(normalize_spec, dict):
            if col in normalize_spec:
                if normalize_spec[col] is None:
                    continue  # Skip normalization
                elif callable(normalize_spec[col]):
                    df[col] = normalize_spec[col](df[col].values)
                else:
                    # Default normalization
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            elif "response" == col and "response" in normalize_spec:
                if normalize_spec["response"] is None:
                    continue
                elif callable(normalize_spec["response"]):
                    df[col] = normalize_spec["response"](df[col].values)
                else:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            else:
                # Default normalization if not specified
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            # Default normalization
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df


def _build_formula(
    df: pd.DataFrame,
    fixed_terms: Sequence[Tuple[str, ...]] = (),
    fixed_interaction_level: int = 1,
    subject_terms: Sequence[Tuple[str, ...]] = (),
    subject_interaction_level: int = 1,
    has_subjects: bool = False,
) -> str:
    """Build the mixed effects model formula."""

    # Get all regressor columns (exclude response and subject)
    regressor_cols = [col for col in df.columns if col not in ["response", "subject"]]

    # Build fixed effects terms
    fixed_effect_terms = []

    # Add intercept
    if fixed_interaction_level >= 0:
        fixed_effect_terms.append("1")

    # Add automatic interaction terms based on level
    if fixed_interaction_level >= 1:
        # Add all single terms
        fixed_effect_terms.extend(regressor_cols)

    if fixed_interaction_level >= 2:
        # Add all pairwise interactions
        for combo in itertools.combinations(regressor_cols, 2):
            fixed_effect_terms.append(":".join(combo))

    if fixed_interaction_level >= 3:
        # Add all three-way interactions
        for combo in itertools.combinations(regressor_cols, 3):
            fixed_effect_terms.append(":".join(combo))

    # Add manually specified fixed terms
    for term_tuple in fixed_terms:
        if len(term_tuple) == 1:
            fixed_effect_terms.append(term_tuple[0])
        else:
            fixed_effect_terms.append(":".join(term_tuple))

    # Remove duplicates while preserving order
    seen = set()
    unique_fixed = []
    for term in fixed_effect_terms:
        if term not in seen:
            seen.add(term)
            unique_fixed.append(term)

    # Build random effects terms if subjects are present
    random_effect_formula = ""
    if has_subjects:
        random_terms = []

        # Add random intercept
        if subject_interaction_level >= 0:
            random_terms.append("1")

        # Add random slopes
        if subject_interaction_level >= 1:
            random_terms.extend(regressor_cols)

        if subject_interaction_level >= 2:
            for combo in itertools.combinations(regressor_cols, 2):
                random_terms.append(":".join(combo))

        # Add manually specified subject terms
        for term_tuple in subject_terms:
            if len(term_tuple) == 1:
                random_terms.append(term_tuple[0])
            else:
                random_terms.append(":".join(term_tuple))

        # Remove duplicates
        seen_random = set()
        unique_random = []
        for term in random_terms:
            if term not in seen_random:
                seen_random.add(term)
                unique_random.append(term)

        if unique_random:
            random_effect_formula = f"~ {' + '.join(unique_random)}"

    # Combine into full formula
    fixed_formula = " + ".join(unique_fixed) if unique_fixed else "1"
    formula = f"response ~ {fixed_formula}"

    return formula, random_effect_formula


def fit_lme(
    response_tree,
    extra_regressors: Optional[Dict[str, Any]] = None,
    fixed_terms: Sequence[Tuple[str, ...]] = (),
    fixed_interaction_level: int = 1,
    subject_axis: Optional[int] = None,
    subject_interaction_level: int = 1,
    subject_terms: Sequence[Tuple[str, ...]] = (),
    normalize: Union[bool, Dict, None] = None,
    regressor_overrides: Optional[Dict] = None,
    **kwargs,
) -> MixedLMResults:
    """
    Fit a linear mixed effects model to PyTree-structured data.

    Parameters
    ----------
    response_tree : PyTree
        PyTree whose leaves are arrays of response values. Tree levels define regressors.
    extra_regressors : dict, optional
        Mapping of regressor names to PyTrees with same structure as response_tree.
    fixed_terms : sequence of tuples
        Additional fixed effect terms to include. Each tuple identifies variables.
    fixed_interaction_level : int
        Level of automatic fixed effect interactions (0=intercept only, 1=main effects,
        2=pairwise, etc.). Default is 1.
    subject_axis : int, optional
        Axis of arrays to treat as subjects for random effects.
    subject_interaction_level : int
        Level of automatic subject effect interactions. Default is 1.
    subject_terms : sequence of tuples
        Additional subject effect interaction terms.
    normalize : bool, dict, or None
        Normalization specification. None (default) normalizes all. False skips normalization.
        Dict maps variable names to callables or None.
    regressor_overrides : dict, optional
        Override inferred regressor values. Maps node types/labels to value nodes.
    **kwargs
        Additional arguments passed to statsmodels MixedLM.fit().

    Returns
    -------
    MixedLMResults
        Fitted mixed effects model results from statsmodels.

    Examples
    --------
    >>> # Create a simple PyTree with two levels of regressors
    >>> from jax_cookbook import LDict
    >>> import jax.numpy as jnp
    >>>
    >>> data = LDict("speed", {
    ...     10: LDict("angle", {
    ...         0: jnp.array([1.2, 1.3, 1.1]),
    ...         45: jnp.array([1.5, 1.6, 1.4]),
    ...     }),
    ...     20: LDict("angle", {
    ...         0: jnp.array([2.1, 2.2, 2.0]),
    ...         45: jnp.array([2.4, 2.5, 2.3]),
    ...     }),
    ... })
    >>>
    >>> # Fit a model with main effects
    >>> result = fit_lme(data, fixed_interaction_level=1)
    >>> print(result.summary())
    """

    # Check for duplicate regressor names
    level_types, level_values = _get_tree_level_info(response_tree)
    regressor_names = [name for name, _ in level_values]
    if extra_regressors:
        regressor_names.extend(extra_regressors.keys())

    if len(regressor_names) != len(set(regressor_names)):
        duplicates = [name for name in regressor_names if regressor_names.count(name) > 1]
        raise ValueError(f"Duplicate regressor names found: {duplicates}")

    # Convert PyTree to DataFrame
    df = _flatten_pytree_to_dataframe(
        response_tree,
        extra_regressors=extra_regressors,
        subject_axis=subject_axis,
        normalize=normalize,
        regressor_overrides=regressor_overrides,
    )

    # Build formula
    has_subjects = "subject" in df.columns
    formula, random_formula = _build_formula(
        df,
        fixed_terms=fixed_terms,
        fixed_interaction_level=fixed_interaction_level,
        subject_terms=subject_terms,
        subject_interaction_level=subject_interaction_level,
        has_subjects=has_subjects,
    )

    # Fit the model
    if has_subjects and random_formula:
        # Mixed effects model with random effects
        model = smf.mixedlm(formula, df, groups=df["subject"], re_formula=random_formula)
    elif has_subjects:
        # Mixed effects model with only random intercept
        model = smf.mixedlm(formula, df, groups=df["subject"])
    else:
        model = smf.ols(formula, df)
        # No subjects specified, fit as fixed effects only (using MixedLM with trivial groups)
        # Create a dummy group column
        # df["_dummy_group"] = 0
        # model = smf.mixedlm(formula, df, groups=df["_dummy_group"])

    # Fit the model with any additional kwargs
    result = model.fit(**kwargs)

    return result


# Additional utility functions


def extract_model_predictions(result: MixedLMResults, original_tree) -> Any:
    """
    Extract model predictions and reshape them back to the original PyTree structure.

    Parameters
    ----------
    result : MixedLMResults
        Fitted model results
    original_tree : PyTree
        Original response tree structure

    Returns
    -------
    PyTree with same structure as original_tree, containing model predictions
    """
    predictions = result.fittedvalues.values

    # Get the original tree structure
    leaves, treedef = jt.flatten(original_tree)

    # Reshape predictions to match original structure
    pred_idx = 0
    new_leaves = []
    for leaf in leaves:
        leaf_size = leaf.size
        leaf_preds = predictions[pred_idx : pred_idx + leaf_size]
        new_leaves.append(leaf_preds.reshape(leaf.shape))
        pred_idx += leaf_size

    return jt.unflatten(treedef, new_leaves)


def extract_random_effects(result: MixedLMResults) -> pd.DataFrame:
    """
    Extract random effects from fitted model.

    Parameters
    ----------
    result : MixedLMResults
        Fitted model results

    Returns
    -------
    pd.DataFrame
        DataFrame containing random effects for each subject
    """
    return result.random_effects
