"""Pipeline contract-native ratchet gate (issue ed225ef).

The C4 pipeline-consolidation gate walks ``src/rlrmp/analysis/pipelines/*.py``
and freezes modules that still bypass the native Feedbax analysis/bundle
contract. New direct durable writers or direct rollout/evaluation reruns must
either be removed or deliberately listed in ``ci/feedbax-contract-suite.toml``
with an owner and rationale.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import tomllib

import pytest
from feedbax.testing import (
    AllowlistDiff,
    AllowlistEntry,
    Scope,
    SiteVisitor,
    diff_allowlist,
    expr_string,
    scan_domain,
)


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "src" / "rlrmp" / "analysis" / "pipelines"
SUITE_MANIFEST_PATH = REPO_ROOT / "ci" / "feedbax-contract-suite.toml"
ANALYSIS_WRITE_TEST_PATH = REPO_ROOT / "tests" / "test_analysis_write_custody.py"

DELETED_OUTPUT_FEEDBACK_MATERIALIZERS = {
    "src/rlrmp/analysis/pipelines/output_feedback_affine_tracker.py",
    "src/rlrmp/analysis/pipelines/output_feedback_linear_recurrent.py",
    "src/rlrmp/analysis/pipelines/output_feedback_phase_modulated_recurrent.py",
    "src/rlrmp/analysis/pipelines/output_feedback_time_constrained.py",
}

DIRECT_EVAL_RERUN_METHODS = {
    "eval_trials",
    "evaluate",
    "rollout",
}
DIRECT_EVAL_RERUN_FUNCTIONS = {
    "rollout_with_kalman_estimator",
    "rollout_with_robust_estimator_policy",
    "run_output_feedback_rollout_recovery",
}
CONTRACT_NATIVE_NAMES = {
    "AbstractAnalysis",
    "AnalysisBundleSpec",
    "AnalysisRunSpec",
    "EvaluationRunManifest",
    "ParentRef",
    "register_analysis_recipe",
}


@dataclass(frozen=True)
class PipelineModuleFacts:
    """Static contract facts for one analysis pipeline module."""

    path: str
    has_contract_native_signal: bool
    durable_write_count: int
    eval_rerun_sites: tuple[str, ...]

    @property
    def has_durable_writer(self) -> bool:
        return self.durable_write_count > 0

    @property
    def has_eval_rerun_bypass(self) -> bool:
        return bool(self.eval_rerun_sites)

    @property
    def requires_allowlist(self) -> bool:
        return self.has_durable_writer or self.has_eval_rerun_bypass

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.has_durable_writer:
            reasons.append("direct_durable_writer")
        if self.has_eval_rerun_bypass:
            reasons.append("direct_eval_rerun")
        return tuple(reasons)


def test_pipeline_contract_scan_is_non_vacuous() -> None:
    facts = _scan_pipeline_tree()

    assert len(facts) >= 3, "pipeline module scan found too few modules"
    assert not any(f.path.endswith("gru_feedback_ablation.py") for f in facts)
    assert len({f.path for f in facts}) == len(facts), "pipeline scan emitted duplicate modules"
    assert any(f.path.endswith("gru_pilot_figures.py") and f.requires_allowlist for f in facts)


def test_pipeline_contract_native_bypass_modules_match_allowlist() -> None:
    diff = _allowlist_diff(_scan_pipeline_tree())
    _assert_no_unlisted_bypass_modules(diff.unlisted)


def test_pipeline_contract_allowlist_has_no_dead_entries() -> None:
    dead = sorted(entry.scope.path for entry in _allowlist_diff(_scan_pipeline_tree()).dead_entries)
    assert not dead, (
        "Pipeline contract-native allowlist names module(s) that no longer "
        f"bypass the contract: {dead}. Remove stale entries; shrinking this "
        "inventory is required."
    )


def test_pipeline_contract_allowlist_entries_carry_owner_and_reason() -> None:
    issue_re = re.compile(r"^[0-9a-f]{7}$")
    entries = _load_allowlist_entries()
    assert entries, "pipeline contract-native allowlist declares zero bypass modules"
    for entry in entries:
        assert issue_re.match(entry.owner), (
            f"Allowlist entry {entry} is missing a 7-character owning issue"
        )
        assert len(entry.reason.strip()) >= 20, f"Allowlist entry {entry} needs a brief reason"
        reasons = entry.key
        assert isinstance(reasons, tuple) and reasons, (
            f"Allowlist entry {entry} needs one or more bypass reasons"
        )
        assert all(reason in {"direct_durable_writer", "direct_eval_rerun"} for reason in reasons)


def test_pipeline_contract_allowlist_excludes_deleted_output_feedback_materializers() -> None:
    allowed_paths = {entry.scope.path for entry in _load_allowlist_entries()}

    forbidden = sorted(allowed_paths & DELETED_OUTPUT_FEEDBACK_MATERIALIZERS)
    assert not forbidden, (
        f"Deleted output-feedback materializers must stay retired, not allowlisted: {forbidden}"
    )


def test_pipeline_contract_negative_canary_flags_unlisted_writer() -> None:
    facts = _scan_source(
        """
from pathlib import Path

def materialize_new_report(output_path: Path) -> None:
    output_path.write_text("new durable report", encoding="utf-8")
""",
        relpath="src/rlrmp/analysis/pipelines/new_report.py",
    )

    with pytest.raises(AssertionError, match="bypass the native pipeline contract"):
        _assert_no_unlisted_bypass_modules(_allowlist_diff([facts], entries=[]).unlisted)


def test_pipeline_contract_negative_canary_flags_unlisted_eval_rerun() -> None:
    facts = _scan_source(
        """
def materialize_new_rollouts(task, model, key):
    return task.eval_trials(model, key)
""",
        relpath="src/rlrmp/analysis/pipelines/new_rollouts.py",
    )

    with pytest.raises(AssertionError, match="bypass the native pipeline contract"):
        _assert_no_unlisted_bypass_modules(_allowlist_diff([facts], entries=[]).unlisted)


def _pipeline_files() -> list[Path]:
    return [
        path
        for path in sorted(PIPELINE_ROOT.glob("*.py"))
        if path.is_file() and path.name != "__init__.py"
    ]


def _scan_pipeline_tree() -> list[PipelineModuleFacts]:
    return scan_domain(
        _pipeline_files(),
        root=REPO_ROOT,
        visitor_factory=_PipelineContractScanner,
    )


def _scan_file(path: Path) -> PipelineModuleFacts:
    relpath = path.relative_to(REPO_ROOT).as_posix()
    return _scan_source(path.read_text(encoding="utf-8"), relpath=relpath)


def _scan_source(source: str, *, relpath: str) -> PipelineModuleFacts:
    tree = ast.parse(source, filename=relpath)
    write_sites = _analysis_write_scanner()._scan_source(source, relpath=relpath)
    scanner = _PipelineContractScanner(relpath=relpath)
    scanner.visit(tree)
    return PipelineModuleFacts(
        path=relpath,
        has_contract_native_signal=scanner.has_contract_native_signal,
        durable_write_count=len(write_sites),
        eval_rerun_sites=tuple(scanner.eval_rerun_sites),
    )


class _PipelineContractScanner(SiteVisitor[PipelineModuleFacts]):
    def __init__(self, *, relpath: str) -> None:
        super().__init__(relpath=relpath)
        self.has_contract_native_signal = False
        self.eval_rerun_sites: list[str] = []
        self._stack: list[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)
        path = REPO_ROOT / self.relpath
        write_sites = (
            _analysis_write_scanner()._scan_source(
                path.read_text(encoding="utf-8"), relpath=self.relpath
            )
            if path.exists()
            else []
        )
        self.sites.append(
            PipelineModuleFacts(
                path=self.relpath,
                has_contract_native_signal=self.has_contract_native_signal,
                durable_write_count=len(write_sites),
                eval_rerun_sites=tuple(self.eval_rerun_sites),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if any(_callable_name(base).split(".")[-1] == "AbstractAnalysis" for base in node.bases):
            self.has_contract_native_signal = True
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in CONTRACT_NATIVE_NAMES:
            self.has_contract_native_signal = True

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in CONTRACT_NATIVE_NAMES:
            self.has_contract_native_signal = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _callable_name(node.func)
        short_name = func_name.split(".")[-1]
        if short_name in CONTRACT_NATIVE_NAMES:
            self.has_contract_native_signal = True
        if self._is_eval_rerun_call(func_name):
            self.eval_rerun_sites.append(f"{self.relpath}:{node.lineno}:{_symbol(self._stack)}")
        self.generic_visit(node)

    @staticmethod
    def _is_eval_rerun_call(func_name: str) -> bool:
        short_name = func_name.split(".")[-1]
        if short_name in DIRECT_EVAL_RERUN_METHODS:
            return True
        return short_name in DIRECT_EVAL_RERUN_FUNCTIONS


def _symbol(stack: list[str]) -> str:
    return ".".join(stack) or "<module>"


def _callable_name(expr: ast.expr) -> str:
    """Render a callable while preserving the gate's historical call unwrapping."""
    while isinstance(expr, ast.Call):
        expr = expr.func
    return expr_string(expr)


def _load_allowlist_entries() -> list[AllowlistEntry[tuple[str, ...]]]:
    manifest = tomllib.loads(SUITE_MANIFEST_PATH.read_text(encoding="utf-8"))
    return [
        AllowlistEntry(
            scope=Scope("file", entry["path"]),
            owner=str(entry.get("owner", "")),
            reason=str(entry.get("reason", "")),
            key=tuple(entry.get("reasons", [])),
        )
        for entry in manifest.get("pipeline_contract_native", {}).get("allowlist", [])
    ]


def _allowlist_diff(
    facts: list[PipelineModuleFacts],
    *,
    entries: list[AllowlistEntry[tuple[str, ...]]] | None = None,
) -> AllowlistDiff[tuple[str, ...], PipelineModuleFacts]:
    return diff_allowlist(
        (fact for fact in facts if fact.requires_allowlist),
        _load_allowlist_entries() if entries is None else entries,
        site_key=lambda fact: fact.reasons,
        site_location=lambda fact: (fact.path, None),
    )


def _assert_no_unlisted_bypass_modules(
    unlisted: tuple[PipelineModuleFacts, ...],
) -> None:
    assert not unlisted, (
        "Pipeline module(s) bypass the native pipeline contract without an "
        f"allowlist entry in {SUITE_MANIFEST_PATH.relative_to(REPO_ROOT)}: "
        f"{[(fact.path, fact.reasons) for fact in unlisted]}. Port the module "
        "onto registered analysis/bundle custody, or add a deliberate shrink-only "
        "entry with owner and rationale."
    )


def _analysis_write_scanner():
    spec = importlib.util.spec_from_file_location(
        "_rlrmp_analysis_write_custody_scan",
        ANALYSIS_WRITE_TEST_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
