"""Native-executor deletion gates for retired training-loop paths."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rlrmp.paths import REPO_ROOT


pytestmark = pytest.mark.feedbax_contract

TRAINING_SOURCE_ROOTS = (
    REPO_ROOT / "src" / "rlrmp" / "train",
    REPO_ROOT / "src" / "rlrmp" / "runtime",
    REPO_ROOT / "scripts",
)
REGISTERED_NATIVE_KERNEL_MODULES = {
    Path("src/rlrmp/train/adaptive_epsilon_native.py"),
    Path("src/rlrmp/train/distillation_native.py"),
    Path("src/rlrmp/train/minimax_native.py"),
    Path("src/rlrmp/train/policy_adversary_native.py"),
}
FORBIDDEN_SPEC_MARKERS = (
    "legacy_runner",
    "legacy_loop_backend",
    "legacy_equivalence_source",
    "legacy_runner_equivalence_reference",
    "native_executor_deferred",
    "_deferred_to",
)


def test_training_optimizer_loops_live_only_in_registered_native_modules() -> None:
    findings = _forbidden_optimizer_loop_findings()

    assert not findings, (
        "Training optimizer-loop body outside registered native kernel modules: "
        + ", ".join(findings)
    )


def test_training_spec_construction_has_no_legacy_executor_markers() -> None:
    findings: list[str] = []
    for path in _python_sources():
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_SPEC_MARKERS:
            if marker in text:
                findings.append(f"{rel}:{marker}")

    assert not findings, "Forbidden native-executor legacy marker(s): " + ", ".join(findings)


def test_native_executor_deletion_gate_negative_canary_flags_optimizer_loop() -> None:
    source = """
def forbidden(model, optimizer, opt_state):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
"""
    tree = ast.parse(source)

    findings = _optimizer_loop_findings_for_tree(
        tree,
        rel_path=Path("src/rlrmp/train/not_native.py"),
    )

    assert findings == ["src/rlrmp/train/not_native.py:2:forbidden"]


def _forbidden_optimizer_loop_findings() -> list[str]:
    findings: list[str] = []
    for path in _python_sources():
        rel = path.relative_to(REPO_ROOT)
        if rel in REGISTERED_NATIVE_KERNEL_MODULES:
            continue
        findings.extend(_optimizer_loop_findings_for_tree(ast.parse(path.read_text()), rel_path=rel))
    return sorted(findings)


def _optimizer_loop_findings_for_tree(tree: ast.AST, *, rel_path: Path) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        has_grad = any(_call_name(call.func).endswith("filter_value_and_grad") for call in calls)
        has_optimizer_update = any(_call_name(call.func).endswith(".update") for call in calls)
        has_apply_updates = any(_call_name(call.func).endswith("apply_updates") for call in calls)
        if has_grad and has_optimizer_update and has_apply_updates:
            findings.append(f"{rel_path}:{node.lineno}:{node.name}")
    return findings


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _python_sources() -> list[Path]:
    paths: list[Path] = []
    for root in TRAINING_SOURCE_ROOTS:
        paths.extend(path for path in root.rglob("*.py") if path.is_file())
    return sorted(paths)
