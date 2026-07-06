"""Native-executor deletion gates for retired training-loop paths."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rlrmp.paths import REPO_ROOT
from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_METHOD_REF,
    GUIDED_DISTILLATION_METHOD_REF,
)


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
REGISTERED_METHOD_CLI_WIRING = {
    CLOSED_LOOP_DISTILLATION_METHOD_REF: {
        "script": Path("scripts/train_closed_loop_distillation.py"),
        "module": Path("src/rlrmp/train/closed_loop_distillation.py"),
        "entrypoint": "main",
    },
    GUIDED_DISTILLATION_METHOD_REF: {
        "script": Path("scripts/train_guided_distillation.py"),
        "module": Path("src/rlrmp/train/guided_distillation.py"),
        "entrypoint": "main",
    },
}
NATIVE_EXECUTOR_CALL = "execute_distillation_training_run_spec_native"
MINIMAX_JITTED_STEP_FUNCTIONS = {
    "_vmapped_gaussian_adversary_ascent",
    "_vmapped_linear_adversary_ascent",
    "_vmapped_controller_descent",
}
ALLOWED_NON_TRAINING_OPTIMIZER_LOOPS = {
    (
        Path("src/rlrmp/train/cs_perturbation_training.py"),
        "run_broad_epsilon_pgd_inner_maximizer",
    ),
    (
        Path("src/rlrmp/train/cs_perturbation_training.py"),
        "_run_finite_broad_epsilon_pgd_inner_maximizer",
    ),
}
ALLOWED_NATIVE_OPTIMIZER_LOOPS = {
    (
        Path("src/rlrmp/train/cs_nominal_gru.py"),
        "_supervised_train_step",
    ),
}


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


def test_registered_distillation_cli_paths_reach_native_executor() -> None:
    findings: list[str] = []
    for method_ref, wiring in REGISTERED_METHOD_CLI_WIRING.items():
        script_path = REPO_ROOT / wiring["script"]
        module_path = REPO_ROOT / wiring["module"]
        script_calls = _module_call_names(ast.parse(script_path.read_text(encoding="utf-8")))
        if "main" not in script_calls:
            findings.append(f"{method_ref}:{wiring['script']}:does_not_call_main")
            continue
        module_tree = ast.parse(module_path.read_text(encoding="utf-8"))
        if not _function_reaches_call(
            module_tree,
            entrypoint=str(wiring["entrypoint"]),
            target_leaf=NATIVE_EXECUTOR_CALL,
        ):
            findings.append(f"{method_ref}:{wiring['module']}:native_executor_unreachable")

    assert not findings, "Registered method CLI does not reach native executor: " + ", ".join(
        findings
    )


def test_minimax_native_step_functions_stay_jitted() -> None:
    module_path = REPO_ROOT / "src" / "rlrmp" / "train" / "minimax_native.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    findings: list[str] = []
    for name in sorted(MINIMAX_JITTED_STEP_FUNCTIONS):
        node = functions.get(name)
        if node is None:
            findings.append(f"{name}:missing")
            continue
        decorators = {_call_name(decorator) for decorator in node.decorator_list}
        if "eqx.filter_jit" not in decorators:
            findings.append(f"{name}:missing_filter_jit")

    assert not findings, "Minimax native step function(s) lost eqx.filter_jit: " + ", ".join(
        findings
    )


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


def test_native_executor_deletion_gate_negative_canary_flags_filter_grad_loop() -> None:
    source = """
def forbidden(model, optimizer, opt_state):
    grads = eqx.filter_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state
"""
    findings = _optimizer_loop_findings_for_tree(
        ast.parse(source),
        rel_path=Path("src/rlrmp/train/not_native.py"),
    )

    assert findings == ["src/rlrmp/train/not_native.py:2:forbidden"]


def test_native_executor_deletion_gate_negative_canary_flags_split_loop_module() -> None:
    source = """
def compute_grad(model):
    return jax.value_and_grad(loss_fn)(model)

def update_model(model, optimizer, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state
"""
    findings = _optimizer_loop_findings_for_tree(
        ast.parse(source),
        rel_path=Path("src/rlrmp/train/not_native.py"),
    )

    assert findings == ["src/rlrmp/train/not_native.py:module:split_optimizer_loop"]


def test_native_executor_deletion_gate_negative_canary_flags_unwired_cli() -> None:
    source = """
def run_legacy(args):
    return legacy_runner(args)

def main(argv=None):
    args = parser.parse_args(argv)
    if args.full_train:
        return run_legacy(args)
    return 0
"""

    assert not _function_reaches_call(
        ast.parse(source),
        entrypoint="main",
        target_leaf=NATIVE_EXECUTOR_CALL,
    )


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
    module_has_grad = False
    module_has_optimizer_update = False
    module_has_apply_updates = False
    allowed_direct_loop = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
        has_grad = any(_is_grad_call(call.func) for call in calls)
        has_optimizer_update = any(_call_name(call.func).endswith(".update") for call in calls)
        has_apply_updates = any(_call_name(call.func).endswith("apply_updates") for call in calls)
        module_has_grad = module_has_grad or has_grad
        module_has_optimizer_update = module_has_optimizer_update or has_optimizer_update
        module_has_apply_updates = module_has_apply_updates or has_apply_updates
        if has_grad and has_optimizer_update and has_apply_updates:
            if (
                (rel_path, node.name) in ALLOWED_NON_TRAINING_OPTIMIZER_LOOPS
                or (rel_path, node.name) in ALLOWED_NATIVE_OPTIMIZER_LOOPS
            ):
                allowed_direct_loop = True
                continue
            findings.append(f"{rel_path}:{node.lineno}:{node.name}")
    if (
        not findings
        and not allowed_direct_loop
        and module_has_grad
        and module_has_optimizer_update
        and module_has_apply_updates
    ):
        findings.append(f"{rel_path}:module:split_optimizer_loop")
    return findings


def _is_grad_call(node: ast.AST) -> bool:
    return _call_leaf(node) in {"filter_grad", "filter_value_and_grad", "grad", "value_and_grad"}


def _call_leaf(node: ast.AST) -> str:
    name = _call_name(node)
    return name.rsplit(".", 1)[-1] if name else ""


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _module_call_names(tree: ast.AST) -> set[str]:
    return {
        _call_leaf(call.func)
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and _call_leaf(call.func)
    }


def _function_reaches_call(tree: ast.AST, *, entrypoint: str, target_leaf: str) -> bool:
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    visited: set[str] = set()

    def visit(function_name: str) -> bool:
        if function_name in visited:
            return False
        visited.add(function_name)
        node = functions.get(function_name)
        if node is None:
            return False
        called = _module_call_names(node)
        if target_leaf in called:
            return True
        return any(visit(called_name) for called_name in called if called_name in functions)

    return visit(entrypoint)


def _python_sources() -> list[Path]:
    paths: list[Path] = []
    for root in TRAINING_SOURCE_ROOTS:
        paths.extend(path for path in root.rglob("*.py") if path.is_file())
    return sorted(paths)
