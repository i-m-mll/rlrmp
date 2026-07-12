"""Controller-feedback descriptor contract tests."""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import jax.numpy as jnp
import numpy as np
import pytest

from rlrmp.eval.gru_diagnostics import (
    RolloutEvaluation,
    summarize_controller_feedback_scales,
)
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    COMPONENT_POSITION,
    COMPONENT_VELOCITY,
    CONTROLLER_FEEDBACK_FORCE_FILTER_ID,
    DESCRIPTOR_PAYLOAD_KEY,
    controller_feedback_axis_index,
    controller_feedback_descriptor_payload,
    resolve_controller_feedback_view,
    resolve_controller_feedback_view_from_gru_input,
)


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_RELATIVE_PATHS = (
    "src/rlrmp/eval/gru_diagnostics.py",
    "src/rlrmp/eval/feedback_ablation.py",
    "src/rlrmp/eval/perturbation_bank.py",
    "src/rlrmp/analysis/pipelines/gru_steady_state_perturbation_bank.py",
    "src/rlrmp/model/feedbax_graph.py",
    "src/rlrmp/train/cs_perturbation_training.py",
)


def test_controller_feedback_descriptor_payloads_cover_4d_and_6d_bases() -> None:
    payload_4d = controller_feedback_descriptor_payload(feedback_dim=4)
    payload_6d = controller_feedback_descriptor_payload(feedback_dim=6)

    assert payload_4d["basis_id"] == "target_relative_delayed_feedback"
    assert payload_4d["component_ids"] == [COMPONENT_POSITION, COMPONENT_VELOCITY]
    assert payload_4d["descriptor_basis_hash"].startswith("sha256:")
    assert payload_6d["basis_id"] == "target_relative_delayed_feedback_plus_force_filter"
    assert payload_6d["component_ids"] == [
        COMPONENT_POSITION,
        COMPONENT_VELOCITY,
        COMPONENT_FORCE_FILTER,
    ]
    force_filter = resolve_controller_feedback_view(payload_6d).component(COMPONENT_FORCE_FILTER)
    assert force_filter.descriptor_id == CONTROLLER_FEEDBACK_FORCE_FILTER_ID
    assert force_filter.slice.start == 4
    assert force_filter.slice.stop == 6
    assert force_filter.units == "N"


def test_resolved_descriptor_view_selects_values_by_component_id() -> None:
    gru_input = jnp.asarray([[[[99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0]]]])

    view = resolve_controller_feedback_view_from_gru_input(gru_input)
    force_filter = view.component(COMPONENT_FORCE_FILTER)

    assert view.feedback_dim == 6
    assert view.start_index == 1
    assert force_filter.absolute_indices == (5, 6)
    np.testing.assert_allclose(np.asarray(force_filter.values), [[[[5.0, 12.0]]]])
    assert controller_feedback_axis_index(COMPONENT_FORCE_FILTER, "y") == 5


def test_controller_feedback_scales_emit_descriptor_payload_and_ids() -> None:
    gru_input = np.zeros((1, 1, 2, 6), dtype=np.float64)
    gru_input[..., :] = np.asarray(
        [[[[1.0, 0.0, 3.0, 4.0, 5.0, 12.0], [2.0, 0.0, 0.0, 6.0, 8.0, 15.0]]]]
    )
    evaluation = RolloutEvaluation(
        position=np.zeros((1, 1, 2, 2)),
        velocity=np.zeros((1, 1, 2, 2)),
        command=np.zeros((1, 1, 2, 2)),
        hidden=np.zeros((1, 1, 2, 2)),
        gru_input=gru_input,
        initial_position=np.zeros((1, 2)),
        initial_velocity=np.zeros((1, 2)),
        target_position=np.zeros((1, 2, 2)),
        dt=0.01,
    )

    summary = summarize_controller_feedback_scales(evaluation, run_id="run_a")

    assert summary[DESCRIPTOR_PAYLOAD_KEY]["basis_id"] == (
        "target_relative_delayed_feedback_plus_force_filter"
    )
    assert (
        summary["descriptor_basis_hash"] == summary[DESCRIPTOR_PAYLOAD_KEY]["descriptor_basis_hash"]
    )
    force_filter = summary["components"][COMPONENT_FORCE_FILTER]
    assert force_filter["descriptor_id"] == CONTROLLER_FEEDBACK_FORCE_FILTER_ID
    assert force_filter["feedback_basis_indices"] == [4, 5]
    assert force_filter["gru_input_indices"] == [4, 5]


def test_feedback_descriptor_ast_scan_has_no_baked_order_or_width_shortcuts() -> None:
    violations: list[str] = []
    for relative_path in SCANNED_RELATIVE_PATHS:
        path = REPO_ROOT / relative_path
        violations.extend(_feedback_descriptor_scan_violations(path, path.read_text()))

    assert violations == []


def test_feedback_descriptor_ast_scan_rejects_known_antipatterns() -> None:
    source = dedent(
        """
        feedback_order = ["pos_x", "pos_y", "vel_x", "vel_y"]
        POSITION_ONLY_MASK = (1.0, 1.0, 0.0, 0.0)
        force_filter = values[..., 4:6]
        feedback_dim = 6 if input_dim >= 6 else 4
        force_filter_rows = (4, 5)
        """
    )

    violations = _feedback_descriptor_scan_violations(Path("synthetic.py"), source)

    assert len(violations) == 5
    assert any("feedback_order" in violation for violation in violations)
    assert any("bare feedback mask" in violation for violation in violations)
    assert any("force/filter slice" in violation for violation in violations)
    assert any("trailing feedback width" in violation for violation in violations)
    assert any("force/filter indices" in violation for violation in violations)


def _feedback_descriptor_scan_violations(path: Path, source: str) -> list[str]:
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        segment = ast.get_source_segment(source, node) or ""
        lineno = getattr(node, "lineno", 0)
        line = source.splitlines()[lineno - 1] if lineno else ""
        if _is_feedback_order_literal(node):
            violations.append(f"{path}:{lineno}: feedback_order literal")
        if _is_bare_feedback_mask(node):
            violations.append(f"{path}:{lineno}: bare feedback mask")
        if _is_force_filter_slice(node, segment, line):
            violations.append(f"{path}:{lineno}: force/filter slice")
        if _is_trailing_width_inference(node, segment):
            violations.append(f"{path}:{lineno}: trailing feedback width inference")
        if _is_force_filter_index_pair(node):
            violations.append(f"{path}:{lineno}: force/filter indices")
    return violations


def _is_feedback_order_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        return any(_name(target) == "feedback_order" for target in node.targets) and isinstance(
            node.value, ast.List | ast.Tuple
        )
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and key.value == "feedback_order"
                and isinstance(value, ast.List | ast.Tuple)
            ):
                return True
    return False


def _is_bare_feedback_mask(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    if not any("MASK" in _name(target) for target in node.targets):
        return False
    return _numeric_sequence(node.value) in {
        (1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
    }


def _is_force_filter_slice(node: ast.AST, segment: str, line: str) -> bool:
    if not isinstance(node, ast.Slice):
        return False
    if _int_constant(node.lower) == 4 and _int_constant(node.upper) == 6:
        return "hex_color" not in segment and "hex_color" not in line
    return False


def _is_trailing_width_inference(node: ast.AST, segment: str) -> bool:
    if not isinstance(node, ast.IfExp):
        return False
    return "input_dim >= 6" in segment and "else 4" in segment


def _is_force_filter_index_pair(node: ast.AST) -> bool:
    return _numeric_sequence(node) in {(4.0, 5.0), (4, 5)}


def _numeric_sequence(node: ast.AST) -> tuple[float, ...] | None:
    if not isinstance(node, ast.List | ast.Tuple):
        return None
    values: list[float] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, int | float):
            return None
        values.append(element.value)
    return tuple(values)


def _int_constant(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _name(node: ast.AST) -> str:
    return node.id if isinstance(node, ast.Name) else ""
