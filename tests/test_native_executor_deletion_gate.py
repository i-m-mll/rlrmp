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
    Path("src/rlrmp/train/distillation_native/executor.py"),
    Path("src/rlrmp/train/distillation_native/guided_kernel.py"),
    Path("src/rlrmp/train/minimax_native/kernels.py"),
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
RETIRED_TRAINING_SCRIPT_PATHS = {
    Path("scripts/train_closed_loop_distillation.py"),
    Path("scripts/train_cs_nominal_gru.py"),
    Path("scripts/train_guided_distillation.py"),
    Path("scripts/train_minimax.py"),
}
NATIVE_EXECUTOR_CALL = "execute_distillation_training_run_spec_native"
MINIMAX_JITTED_STEP_FUNCTIONS = {
    "_vmapped_gaussian_adversary_ascent",
    "_vmapped_linear_adversary_ascent",
    "_vmapped_controller_descent",
}
RETIRED_GUIDED_CHECKPOINT_APIS = {
    "GuidedDistillationTrainingState",
    "latest_checkpoint_path",
    "save_training_checkpoint",
    "load_latest_checkpoint",
}
RETIRED_TRAINER_MODULES = {
    Path("src/rlrmp/train/minimax.py"),
    Path("src/rlrmp/train/distillation.py"),
    Path("src/rlrmp/train/guided_distillation.py"),
    Path("src/rlrmp/train/closed_loop_distillation.py"),
}
RETIRED_RUNTIME_DEFINITIONS = {
    "build_parser",
    "_build_parser",
    "main",
    "build_distillation_spec",
    "build_closed_loop_distillation_spec",
    "run_guided_distillation_training",
    "run_closed_loop_distillation_training_native",
    "write_run_spec",
    "_atomic_write_json",
    "legacy_cli_args_to_minimax_config",
    "minimax_config_namespace",
    "_legacy_minimax_run_spec_payload",
    "_rlrmp_minimax_extension_payload",
    "_minimax_method_payload",
}
ALLOWED_NON_TRAINING_OPTIMIZER_LOOPS = {
    (
        Path("src/rlrmp/train/cs_perturbation_training.py"),
        "_run_broad_epsilon_pgd_ascent",
    ),
}
ALLOWED_NON_TRAINING_OPTIMIZER_MODULES = {
    Path("src/rlrmp/train/broad_epsilon_training.py"),
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


def test_retired_training_script_paths_stay_deleted() -> None:
    revived = sorted(str(path) for path in RETIRED_TRAINING_SCRIPT_PATHS if (REPO_ROOT / path).exists())

    assert not revived, "Retired training script path(s) reappeared: " + ", ".join(revived)


def test_executable_train_script_surfaces_do_not_reaccrete() -> None:
    revived = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "scripts").glob("train_*.py")
        if path.is_file()
    )

    assert not revived, "Executable scripts/train_*.py surface(s) reappeared: " + ", ".join(revived)


def test_minimax_native_step_functions_stay_jitted() -> None:
    module_path = REPO_ROOT / "src" / "rlrmp" / "train" / "minimax_native" / "kernels.py"
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


def test_retired_guided_checkpoint_runtime_apis_stay_deleted() -> None:
    module_path = REPO_ROOT / "src" / "rlrmp" / "train" / "distillation_native" / "guided_kernel.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }

    revived = sorted(definitions & RETIRED_GUIDED_CHECKPOINT_APIS)

    assert not revived, "Retired guided checkpoint API(s) reappeared: " + ", ".join(revived)


def test_minimax_checkpoint_slots_are_executor_owned() -> None:
    minimax_path = REPO_ROOT / "src" / "rlrmp" / "train" / "minimax_native" / "method.py"
    slots_path = REPO_ROOT / "src" / "rlrmp" / "train" / "executor" / "slots.py"
    minimax_tree = ast.parse(minimax_path.read_text(encoding="utf-8"))
    slots_tree = ast.parse(slots_path.read_text(encoding="utf-8"))

    minimax_calls = _module_call_names(minimax_tree)
    slots_defs = {
        node.name
        for node in ast.walk(slots_tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    assert "minimax_checkpoint_slot_specs" in slots_defs
    assert "minimax_checkpoint_slot_specs" in minimax_calls
    assert "CheckpointSlotSpec" not in _module_call_names(minimax_tree)


def test_pre_native_trainer_modules_are_deleted_not_reexported() -> None:
    revived = sorted(str(path) for path in RETIRED_TRAINER_MODULES if (REPO_ROOT / path).exists())
    assert not revived, "Retired trainer module(s) reappeared: " + ", ".join(revived)


def test_native_packages_do_not_hide_relocated_authoring_or_writer_bodies() -> None:
    caps = {
        Path("src/rlrmp/train/minimax_native/method.py"): 900,
        Path("src/rlrmp/train/distillation_native/guided_kernel.py"): 600,
        Path("src/rlrmp/train/distillation_native/closed_loop_kernel.py"): 525,
    }
    findings: list[str] = []
    for rel_path, line_cap in caps.items():
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        tree = ast.parse(text)
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        for name in sorted(definitions & RETIRED_RUNTIME_DEFINITIONS):
            findings.append(f"{rel_path}:{name}")
        if len(text.splitlines()) > line_cap:
            findings.append(f"{rel_path}:exceeds_{line_cap}_line_cap")
    assert not findings, "Retired runtime/authoring body moved under native package: " + ", ".join(
        findings
    )


def test_distillation_kernel_dependency_is_acyclic() -> None:
    kernel_paths = (
        Path("src/rlrmp/train/distillation_native/guided_kernel.py"),
        Path("src/rlrmp/train/distillation_native/closed_loop_kernel.py"),
    )
    findings: list[str] = []
    for rel_path in kernel_paths:
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        if "rlrmp.train.distillation_native.executor" in imported:
            findings.append(f"{rel_path}:imports_executor")
    assert not findings, "Distillation kernel/executor cycle reappeared: " + ", ".join(findings)


def test_scripts_contain_no_script_local_checkpoint_or_optimizer_runtime() -> None:
    forbidden = {
        "_save_adversarial_checkpoint",
        "_load_adversarial_checkpoint",
        "_write_warmup_boundary_checkpoint",
        "_write_final_minimax_custody_transaction",
        "_eval_trials_streaming",
    }
    findings: list[str] = []
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        rel_path = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        findings.extend(f"{rel_path}:{name}" for name in sorted(definitions & forbidden))
        findings.extend(_optimizer_loop_findings_for_tree(tree, rel_path=rel_path))

    assert not findings, "Script-local checkpoint/optimizer runtime reappeared: " + ", ".join(
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
        if rel in REGISTERED_NATIVE_KERNEL_MODULES or rel in ALLOWED_NON_TRAINING_OPTIMIZER_MODULES:
            continue
        findings.extend(
            _optimizer_loop_findings_for_tree(ast.parse(path.read_text()), rel_path=rel)
        )
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
        if (
            has_optimizer_update
            and has_apply_updates
            and (rel_path, node.name) in ALLOWED_NON_TRAINING_OPTIMIZER_LOOPS
        ):
            allowed_direct_loop = True
            continue
        if has_grad and has_optimizer_update and has_apply_updates:
            if (rel_path, node.name) in ALLOWED_NATIVE_OPTIMIZER_LOOPS:
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
