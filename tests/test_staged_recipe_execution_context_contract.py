"""Structural contract for RLRMP staged evaluation and analysis recipes."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_type_hints

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
CHILD_RESULT_PREFIX = "RLRMP_STAGED_CONTRACT_JSON="

EXPECTED_RLRMP_EVALUATION_TYPES = (
    "rlrmp.eval.broad_epsilon",
    "rlrmp.eval.center_out_ensemble",
    "rlrmp.eval.delayed_reach_bank",
    "rlrmp.eval.feedback_ablation",
    "rlrmp.eval.gru_diagnostics",
    "rlrmp.eval.linear_recurrent_augmented_reference",
    "rlrmp.eval.output_feedback_rollout_recovery",
    "rlrmp.eval.perturbation_response_bank",
    "rlrmp.eval.worst_case_epsilon",
    "rlrmp.sisu_spectrum_evaluation",
    "rlrmp.standard_matrix_evaluation",
)

EXPECTED_RLRMP_ANALYSIS_TYPES = (
    "rlrmp.analysis.broad_epsilon_attribution",
    "rlrmp.analysis.objective_comparator",
    "rlrmp.analysis.worst_case_epsilon",
    "rlrmp.certificate.gru_standard",
    "rlrmp.certificate.standard",
    "rlrmp.diagnostic.gru_evaluation",
    "rlrmp.diagnostic.policy_local",
    "rlrmp.diagnostic.recurrent_jacobian",
    "rlrmp.feedback_ablation",
    "rlrmp.feedback_quality.evaluation_diagnostics",
    "rlrmp.feedback_quality.feedback_ablation",
    "rlrmp.feedback_quality.objective_comparator",
    "rlrmp.feedback_quality.perturbation_calibration",
    "rlrmp.feedback_quality.perturbation_response",
    "rlrmp.feedback_quality.response_norm_plots",
    "rlrmp.feedback_quality_lens",
    "rlrmp.history_payload",
    "rlrmp.map_error_decomposition",
    "rlrmp.output_feedback_bridge.rollout_recovery",
    "rlrmp.perturbation_bank_aggregate",
    "rlrmp.perturbation_class_response",
    "rlrmp.response_norm_comparison",
    "rlrmp.robustness_margin_sidecar",
    "rlrmp.robustness_phenotype",
    "rlrmp.scalar_diagnostic_payload",
    "rlrmp.sisu_robustification_comparison",
    "rlrmp.sisu_spectrum",
    "rlrmp.standard_matrix",
    "rlrmp.training_diagnostics_summary",
)


CHILD_PROGRAM = textwrap.dedent(
    f"""
    import inspect
    import json
    import sys
    from pathlib import Path
    from typing import get_type_hints

    from feedbax.analysis import (
        EMPTY_STAGED_EXECUTION_CONTEXT,
        StagedExecutionContext,
        get_evaluation_recipe,
        registered_evaluation_recipes,
    )
    from feedbax.analysis.evaluation import execute_evaluation_run_spec
    from feedbax.analysis.specs import (
        execute_analysis_run_spec,
        get_analysis_recipe,
        registered_analysis_types,
    )
    from feedbax.contracts.manifest import (
        AnalysisRunSpec,
        EvaluationRunSpec,
        evaluation_states_cache_path,
    )
    from feedbax.plugins.registry import ExperimentRegistry

    import rlrmp
    from rlrmp.runtime.studio_records import (
        STUDIO_DEFAULT_EVALUATION_TYPE,
        register_rlrmp_studio_recipes,
    )


    def assert_exact_staged_recipe_signature(identity, recipe):
        signature = inspect.signature(recipe)
        parameters = tuple(signature.parameters.values())
        assert len(parameters) == 4, (
            f"{{identity}} must expose exactly four parameters; got {{signature}}"
        )
        assert all(
            parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        ), f"{{identity}} must not use keyword-only or variadic parameters: {{signature}}"
        assert all(
            parameter.default is inspect.Parameter.empty for parameter in parameters
        ), f"{{identity}} must require all four positional parameters: {{signature}}"
        context_parameter = parameters[3]
        annotations = get_type_hints(recipe)
        assert annotations.get(context_parameter.name) is StagedExecutionContext, (
            f"{{identity}} fourth parameter must use public StagedExecutionContext: "
            f"{{signature}}"
        )


    root = Path(sys.argv[1])
    expected = json.loads(sys.argv[2])

    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    package_names = registry.get_package_names()
    assert package_names == ["rlrmp"]

    evaluation_types = [
        identity
        for identity in registered_evaluation_recipes()
        if identity.startswith("rlrmp.")
    ]
    analysis_types = [
        identity
        for identity in registered_analysis_types()
        if identity.startswith("rlrmp.")
    ]
    assert evaluation_types == expected["evaluation_types"]
    assert analysis_types == expected["analysis_types"]

    for identity in evaluation_types:
        assert_exact_staged_recipe_signature(identity, get_evaluation_recipe(identity))
    for identity in analysis_types:
        assert_exact_staged_recipe_signature(identity, get_analysis_recipe(identity))

    register_rlrmp_studio_recipes(replace=True)
    studio_recipe = get_evaluation_recipe(STUDIO_DEFAULT_EVALUATION_TYPE)
    assert studio_recipe.__module__ == "rlrmp.runtime.studio_records"
    assert_exact_staged_recipe_signature(STUDIO_DEFAULT_EVALUATION_TYPE, studio_recipe)

    evaluation_manifest, evaluation_path = execute_evaluation_run_spec(
        EvaluationRunSpec(evaluation_type="rlrmp.standard_matrix_evaluation"),
        root=root,
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        force=True,
    )
    evaluation_states_path = evaluation_states_cache_path(
        evaluation_manifest.id,
        root=root,
    )
    assert evaluation_manifest.status == "completed"
    assert evaluation_manifest.summary_metrics["standard_matrix_cells"] == 0
    assert evaluation_path.is_file()
    assert evaluation_states_path.is_file()

    analysis_manifest, analysis_path = execute_analysis_run_spec(
        AnalysisRunSpec(
            analysis_type="rlrmp.standard_matrix",
            params={{"requested_outputs": ["summary_metrics"]}},
        ),
        root=root,
        execution_context=EMPTY_STAGED_EXECUTION_CONTEXT,
        force=True,
    )
    assert analysis_manifest.status == "completed"
    assert analysis_manifest.summary_metrics["analysis_count"] == 1
    assert analysis_path.is_file()

    print(
        {CHILD_RESULT_PREFIX!r}
        + json.dumps(
            {{
                "package_names": package_names,
                "evaluation_types": evaluation_types,
                "analysis_types": analysis_types,
                "studio_recipe_module": studio_recipe.__module__,
                "evaluation_status": evaluation_manifest.status,
                "evaluation_manifest_materialized": evaluation_path.is_file(),
                "evaluation_states_materialized": evaluation_states_path.is_file(),
                "standard_matrix_cells": evaluation_manifest.summary_metrics[
                    "standard_matrix_cells"
                ],
                "analysis_status": analysis_manifest.status,
                "analysis_manifest_materialized": analysis_path.is_file(),
                "analysis_count": analysis_manifest.summary_metrics["analysis_count"],
            }},
            sort_keys=True,
        )
    )
    """
)


class _NegativeCanaryContext:
    """Local annotation target that keeps negative canaries Feedbax-free."""


def _assert_exact_staged_recipe_signature(
    identity: str,
    recipe: Callable[..., Any],
) -> None:
    signature = inspect.signature(recipe)
    parameters = tuple(signature.parameters.values())

    assert len(parameters) == 4, f"{identity} must expose exactly four parameters; got {signature}"
    assert all(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    ), f"{identity} must not use keyword-only or variadic parameters: {signature}"
    assert all(parameter.default is inspect.Parameter.empty for parameter in parameters), (
        f"{identity} must require all four positional parameters: {signature}"
    )

    execution_context_parameter = parameters[3]
    resolved_annotations = get_type_hints(recipe)
    assert resolved_annotations.get(execution_context_parameter.name) is _NegativeCanaryContext


def _old_three_argument_recipe(_spec: Any, _root: Path, _inputs: Any) -> None:
    return None


def _variadic_recipe(
    _spec: Any,
    _root: Path,
    _inputs: Any,
    _execution_context: _NegativeCanaryContext,
    *args: Any,
) -> None:
    return None


def _keyword_variadic_recipe(
    _spec: Any,
    _root: Path,
    _inputs: Any,
    _execution_context: _NegativeCanaryContext,
    **kwargs: Any,
) -> None:
    return None


def _defaulted_context_recipe(
    _spec: Any,
    _root: Path,
    _inputs: Any,
    _execution_context: _NegativeCanaryContext = _NegativeCanaryContext(),
) -> None:
    return None


@pytest.mark.parametrize(
    "recipe",
    (
        _old_three_argument_recipe,
        _variadic_recipe,
        _keyword_variadic_recipe,
        _defaulted_context_recipe,
    ),
)
def test_staged_signature_guard_rejects_arity_evasions(
    recipe: Callable[..., Any],
) -> None:
    with pytest.raises(AssertionError):
        _assert_exact_staged_recipe_signature("rlrmp.test.invalid", recipe)


def test_registered_staged_recipe_contract_isolated_in_child_process(
    tmp_path: Path,
) -> None:
    expected = {
        "evaluation_types": list(EXPECTED_RLRMP_EVALUATION_TYPES),
        "analysis_types": list(EXPECTED_RLRMP_ANALYSIS_TYPES),
    }
    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SRC_ROOT), inherited_pythonpath) if path
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_PROGRAM,
                str(tmp_path),
                json.dumps(expected, sort_keys=True),
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "isolated staged-recipe contract child failed\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}",
            pytrace=False,
        )

    payload_lines = [
        line.removeprefix(CHILD_RESULT_PREFIX)
        for line in result.stdout.splitlines()
        if line.startswith(CHILD_RESULT_PREFIX)
    ]
    assert len(payload_lines) == 1, (
        "isolated staged-recipe contract child returned no unique result payload\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    payload = json.loads(payload_lines[0])

    assert payload == {
        "analysis_count": 1,
        "analysis_manifest_materialized": True,
        "analysis_status": "completed",
        "analysis_types": list(EXPECTED_RLRMP_ANALYSIS_TYPES),
        "evaluation_manifest_materialized": True,
        "evaluation_states_materialized": True,
        "evaluation_status": "completed",
        "evaluation_types": list(EXPECTED_RLRMP_EVALUATION_TYPES),
        "package_names": ["rlrmp"],
        "standard_matrix_cells": 0,
        "studio_recipe_module": "rlrmp.runtime.studio_records",
    }
