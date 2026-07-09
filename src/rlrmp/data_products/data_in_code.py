"""Destination-based data-in-code scanner.

This gate flags common shapes where run, evaluation, or analysis parameters are
encoded directly in Python source instead of living on governed spec/data
surfaces. It deliberately uses AST structure rather than regexes so findings can
be keyed by the object that owns the parameterization.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from rlrmp.data_products import lint as empirical_lint

__all__ = [
    "BASELINE_RELPATH",
    "DATA_IN_CODE_ALLOWLIST",
    "DATA_IN_CODE_DETECTORS",
    "DataInCodeFinding",
    "DataInCodePolicyError",
    "DATA_IN_CODE_POLICY",
    "HP_NAME_LEXICON",
    "baseline_path",
    "default_spec_constructor_names",
    "load_baseline",
    "scan_source",
    "scan_tree",
    "policy_for_finding",
    "validate_findings",
    "violations",
    "write_baseline",
]

BASELINE_RELPATH = "ci/data_in_code_baseline.json"

DATA_IN_CODE_DETECTORS = (
    "argv_rows",
    "spec_flow",
    "default_bundle",
    "hp_constant",
    "empirical_table",
)

DATA_IN_CODE_POLICY: dict[tuple[str, str], str] = {
    ("argv_rows", "src"): "ratchet",
    ("argv_rows", "scripts"): "ratchet",
    ("argv_rows", "results_scripts"): "ratchet",
    ("spec_flow", "src"): "ratchet",
    ("spec_flow", "scripts"): "ratchet",
    ("spec_flow", "results_scripts"): "advisory",
    ("default_bundle", "src"): "ratchet",
    ("default_bundle", "scripts"): "ratchet",
    ("default_bundle", "results_scripts"): "advisory",
    ("hp_constant", "src"): "ratchet",
    ("hp_constant", "scripts"): "ratchet",
    ("hp_constant", "results_scripts"): "advisory",
    ("empirical_table", "src"): "enforced",
    ("empirical_table", "scripts"): "enforced",
    ("empirical_table", "results_scripts"): "ratchet",
}

HP_NAME_LEXICON = (
    "ALPHA",
    "AMPLITUDE",
    "BATCH",
    "BUDGET",
    "CLIP",
    "DT",
    "EPSILON",
    "GAMMA",
    "HIDDEN_SIZE",
    "LEARNING_RATE",
    "LR",
    "NOISE",
    "N_BATCHES",
    "N_REPLICATES",
    "N_ROLLOUT",
    "N_STEPS",
    "N_TRIALS",
    "RADIUS",
    "SCALE",
    "SEED",
    "STD",
    "WARMUP",
    "WEIGHT",
)

_NUMERIC_STRING_RE = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")
_DIMENSION_NAME_TOKENS = frozenset({"DIM", "DIMS", "DIMENSION", "DIMENSIONS", "SHAPE"})
_SPEC_CONSTRUCTOR_SEEDS = frozenset(
    {
        "EvaluationRunSpec",
        "ExtractionProductSpec",
        "LossTermSpec",
        "LrScheduleSpec",
        "MatrixRow",
        "OptimizerSpec",
        "OverridePatch",
        "TrainingRunMatrixSpec",
        "TrainingRunSpec",
    }
)


class DataInCodePolicyError(RuntimeError):
    """Raised when the baseline/allowlist policy would be violated."""


@dataclass(frozen=True)
class DataInCodeFinding:
    """A source object that contains destination-significant literal data."""

    relpath: str
    lineno: int
    qualname: str
    detector: str
    tier: str
    summary: str

    @property
    def key(self) -> str:
        return f"{self.relpath}::{self.qualname}::{self.detector}"


def _empirical_allowlist() -> dict[str, str]:
    return {
        f"{relpath_and_name}::empirical_table": rationale
        for relpath_and_name, rationale in empirical_lint.ALLOWLIST.items()
    }


DATA_IN_CODE_ALLOWLIST: dict[str, str] = {
    **_empirical_allowlist(),
}


def default_spec_constructor_names(repo_root: Path | None = None) -> frozenset[str]:
    """Return the spec-constructor vocabulary used by the ``spec_flow`` detector."""

    names = set(_SPEC_CONSTRUCTOR_SEEDS)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    for relpath in (
        "src/rlrmp/runtime/training_run_specs.py",
        "src/rlrmp/train/minimax.py",
        "src/rlrmp/train/guided_distillation.py",
    ):
        path = repo_root / relpath
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_spec_builder_name(node.name):
                names.add(node.name)
    return frozenset(sorted(names))


def scan_source(
    text: str,
    relpath: str,
    *,
    constructor_names: Iterable[str] | None = None,
    include_empirical: bool = True,
) -> list[DataInCodeFinding]:
    """Scan one Python source string for destination-based data-in-code."""

    tree = ast.parse(text)
    tier = tier_for_relpath(relpath)
    if tier is None:
        return []
    if constructor_names is None:
        constructor_names = default_spec_constructor_names()
    visitor = _DataInCodeVisitor(
        relpath=relpath,
        tier=tier,
        constructor_names=frozenset(constructor_names),
    )
    visitor.visit(tree)
    findings = visitor.findings()
    if include_empirical:
        findings.extend(_empirical_findings(text, relpath, tier))
    return sorted(findings, key=lambda finding: (finding.key, finding.lineno))


def scan_tree(repo_root: Path) -> list[DataInCodeFinding]:
    """Scan ``src/``, ``scripts/``, and ``results/*/scripts/`` under ``repo_root``."""

    repo_root = repo_root.resolve()
    constructor_names = default_spec_constructor_names(repo_root)
    findings: list[DataInCodeFinding] = []
    for path in _scan_paths(repo_root):
        relpath = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            findings.extend(
                scan_source(text, relpath, constructor_names=constructor_names)
            )
        except SyntaxError:
            continue
    return sorted(findings, key=lambda finding: (finding.key, finding.lineno))


def violations(repo_root: Path) -> list[DataInCodeFinding]:
    """Return live-tree findings that are neither allowlisted nor baselined."""

    baseline = set(load_baseline(repo_root))
    return [
        finding
        for finding in scan_tree(repo_root)
        if _is_enforced_or_ratchet(finding)
        and finding.key not in DATA_IN_CODE_ALLOWLIST
        and finding.key not in baseline
    ]


def validate_findings(repo_root: Path) -> None:
    """Assert the live tree exactly matches the committed baseline plus allowlist."""

    findings = scan_tree(repo_root)
    findings_by_key = {finding.key: finding for finding in findings}
    allowlist_keys = set(DATA_IN_CODE_ALLOWLIST)
    baseline_keys = set(load_baseline(repo_root))

    unknown_allowlist = sorted(allowlist_keys - set(findings_by_key))
    if unknown_allowlist:
        raise DataInCodePolicyError(
            "stale data-in-code allowlist entries: " + ", ".join(unknown_allowlist)
        )

    weak_rationales = [
        key
        for key, rationale in DATA_IN_CODE_ALLOWLIST.items()
        if not isinstance(rationale, str) or len(rationale.strip()) < 40
    ]
    if weak_rationales:
        raise DataInCodePolicyError(
            "data-in-code allowlist entries lack rationale: " + ", ".join(weak_rationales)
        )

    current_unallowlisted = {
        key
        for key, finding in findings_by_key.items()
        if _is_enforced_or_ratchet(finding) and key not in allowlist_keys
    }
    added = sorted(current_unallowlisted - baseline_keys)
    stale = sorted(baseline_keys - current_unallowlisted)
    if added:
        raise DataInCodePolicyError("new data-in-code findings: " + ", ".join(added))
    if stale:
        raise DataInCodePolicyError("stale data-in-code baseline keys: " + ", ".join(stale))


def baseline_path(repo_root: Path) -> Path:
    """Return the committed data-in-code baseline path."""

    return repo_root / BASELINE_RELPATH


def load_baseline(repo_root: Path) -> list[str]:
    """Load the committed shrink-only baseline."""

    path = baseline_path(repo_root)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise DataInCodePolicyError(f"{BASELINE_RELPATH} must be a JSON list of strings")
    return sorted(data)


def write_baseline(repo_root: Path, *, allow_growth: bool = False) -> list[str]:
    """Write the current non-allowlisted findings as a shrink-only baseline."""

    current = sorted(
        finding.key
        for finding in scan_tree(repo_root)
        if _is_enforced_or_ratchet(finding) and finding.key not in DATA_IN_CODE_ALLOWLIST
    )
    path = baseline_path(repo_root)
    existing = load_baseline(repo_root)
    if path.exists() and len(current) > len(existing) and not allow_growth:
        raise DataInCodePolicyError(
            f"refusing to grow {BASELINE_RELPATH}: {len(existing)} -> {len(current)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return current


def tier_for_relpath(relpath: str) -> str | None:
    """Return the enforcement tier for a repo-relative Python path."""

    parts = Path(relpath).parts
    if not parts or parts[0] == "tests":
        return None
    if parts[0] == "src":
        return "src"
    if parts[0] == "scripts":
        return "scripts"
    if len(parts) >= 3 and parts[0] == "results" and parts[2] == "scripts":
        return "results_scripts"
    return None


def policy_for_finding(finding: DataInCodeFinding) -> str:
    """Return ``enforced``, ``ratchet``, or ``advisory`` for a finding."""

    try:
        return DATA_IN_CODE_POLICY[(finding.detector, finding.tier)]
    except KeyError as error:
        raise DataInCodePolicyError(
            f"no data-in-code policy for {finding.detector!r} in tier {finding.tier!r}"
        ) from error


def _is_enforced_or_ratchet(finding: DataInCodeFinding) -> bool:
    return policy_for_finding(finding) in {"enforced", "ratchet"}


class _DataInCodeVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        relpath: str,
        tier: str,
        constructor_names: frozenset[str],
    ) -> None:
        self.relpath = relpath
        self.tier = tier
        self.constructor_names = constructor_names
        self._scope: list[str] = []
        self._scope_kinds: list[str] = []
        self._parents: list[ast.AST] = []
        self._module_or_class_assignment: list[str] = []
        self._findings: dict[str, DataInCodeFinding] = {}

    def findings(self) -> list[DataInCodeFinding]:
        return list(self._findings.values())

    def visit(self, node: ast.AST) -> None:  # noqa: D102
        self._parents.append(node)
        try:
            super().visit(node)
        finally:
            self._parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._scan_default_bundle_function(node)
        self._scope.append(node.name)
        self._scope_kinds.append("function")
        self.generic_visit(node)
        self._scope.pop()
        self._scope_kinds.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._scan_class_default_bundle(node)
        self._scope.append(node.name)
        self._scope_kinds.append("class")
        self.generic_visit(node)
        self._scope.pop()
        self._scope_kinds.pop()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self._scan_hp_constant(node)
        assignment_name = self._module_or_class_assignment_name(node)
        if assignment_name is None:
            self.generic_visit(node)
            return
        self._module_or_class_assignment.append(assignment_name)
        self.generic_visit(node)
        self._module_or_class_assignment.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self._scan_hp_constant(node)
        assignment_name = self._module_or_class_assignment_name(node)
        if assignment_name is None:
            self.generic_visit(node)
            return
        self._module_or_class_assignment.append(assignment_name)
        self.generic_visit(node)
        self._module_or_class_assignment.pop()

    def visit_List(self, node: ast.List) -> None:  # noqa: N802
        self._scan_argv_literal(node)
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:  # noqa: N802
        self._scan_argv_literal(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._scan_spec_flow(node)
        self.generic_visit(node)

    def _scan_argv_literal(self, node: ast.List | ast.Tuple) -> None:
        if self._inside_argparse_add_argument():
            return
        has_flag = any(
            isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and element.value.startswith("--")
            for element in node.elts
        )
        has_numeric = any(_is_numeric_constant(element) for element in node.elts)
        if has_flag and has_numeric:
            self._emit(
                detector="argv_rows",
                lineno=node.lineno,
                summary="CLI argv row literal carries numeric run parameters",
            )

    def _scan_spec_flow(self, node: ast.Call) -> None:
        if self._inside_model_class_definition():
            return
        name = _call_name(node.func)
        if name not in self.constructor_names:
            return
        if any(keyword.arg is not None and _literal_contains_numeric(keyword.value)
               for keyword in node.keywords):
            self._emit(
                detector="spec_flow",
                lineno=node.lineno,
                summary=f"{name} keyword literal carries run/eval/analysis parameters",
            )

    def _scan_default_bundle_function(self, node: ast.FunctionDef) -> None:
        local_dicts: dict[str, ast.Dict] = {}
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child is not node:
                    continue
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Dict):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        local_dicts[target.id] = child.value
            elif (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and isinstance(child.value, ast.Dict)
            ):
                local_dicts[child.target.id] = child.value
            elif isinstance(child, ast.Return):
                dict_node: ast.Dict | None = None
                if isinstance(child.value, ast.Dict):
                    dict_node = child.value
                elif isinstance(child.value, ast.Name):
                    dict_node = local_dicts.get(child.value.id)
                if dict_node is not None and _is_default_bundle_dict(dict_node):
                    self._emit_at_qualname(
                        qualname=self._nested_name(node.name),
                        detector="default_bundle",
                        lineno=child.lineno,
                        summary="function returns a hyperparameter-like numeric bundle",
                    )

    def _scan_class_default_bundle(self, node: ast.ClassDef) -> None:
        if self.relpath.startswith("src/rlrmp/runtime/") or "schema" in self.relpath:
            return
        numeric_hp_defaults = 0
        for stmt in node.body:
            name: str | None = None
            value: ast.AST | None = None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                value = stmt.value
            elif isinstance(stmt, ast.Assign):
                names = [target.id for target in stmt.targets if isinstance(target, ast.Name)]
                if len(names) == 1:
                    name = names[0]
                    value = stmt.value
            if name is None or value is None:
                continue
            if _hp_name_matches(name) and _literal_contains_numeric(value):
                numeric_hp_defaults += 1
        if numeric_hp_defaults >= 3:
            self._emit_at_qualname(
                qualname=self._nested_name(node.name),
                detector="default_bundle",
                lineno=node.lineno,
                summary="class defaults contain a scattered hyperparameter bundle",
            )

    def _scan_hp_constant(self, node: ast.Assign | ast.AnnAssign) -> None:
        if self._scope_kinds and self._scope_kinds[-1] == "function":
            return
        names = _assignment_names(node)
        value = _assignment_value(node)
        if value is None or not _is_literal_tree(value):
            return
        for name in names:
            if _hp_name_matches(name):
                self._emit_at_qualname(
                    qualname=self._nested_name(name),
                    detector="hp_constant",
                    lineno=node.lineno,
                    summary="module/class hyperparameter-like constant is literal-backed",
                )

    def _module_or_class_assignment_name(
        self,
        node: ast.Assign | ast.AnnAssign,
    ) -> str | None:
        if self._scope_kinds and self._scope_kinds[-1] == "function":
            return None
        names = _assignment_names(node)
        if len(names) != 1:
            return None
        value = _assignment_value(node)
        if value is None or not isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return None
        return self._nested_name(names[0])

    def _inside_argparse_add_argument(self) -> bool:
        for parent in reversed(self._parents[:-1]):
            if not isinstance(parent, ast.Call):
                continue
            if _call_name(parent.func) == "add_argument":
                return True
        return False

    def _inside_model_class_definition(self) -> bool:
        return bool(self._scope_kinds and self._scope_kinds[-1] == "class")

    def _current_qualname(self) -> str:
        if self._module_or_class_assignment:
            return self._module_or_class_assignment[-1]
        if self._scope:
            return ".".join(self._scope)
        return "<module>"

    def _nested_name(self, name: str) -> str:
        if self._scope:
            return ".".join((*self._scope, name))
        return name

    def _emit(self, *, detector: str, lineno: int, summary: str) -> None:
        self._emit_at_qualname(
            qualname=self._current_qualname(),
            detector=detector,
            lineno=lineno,
            summary=summary,
        )

    def _emit_at_qualname(
        self,
        *,
        qualname: str,
        detector: str,
        lineno: int,
        summary: str,
    ) -> None:
        finding = DataInCodeFinding(
            relpath=self.relpath,
            lineno=lineno,
            qualname=qualname,
            detector=detector,
            tier=self.tier,
            summary=summary,
        )
        self._findings.setdefault(finding.key, finding)


def _scan_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (repo_root / "src", repo_root / "scripts"):
        if root.exists():
            paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    results_root = repo_root / "results"
    if results_root.exists():
        for scripts_root in sorted(results_root.glob("*/scripts")):
            paths.extend(
                path for path in scripts_root.rglob("*.py") if "__pycache__" not in path.parts
            )
    return sorted(paths)


def _empirical_findings(text: str, relpath: str, tier: str) -> list[DataInCodeFinding]:
    return [
        DataInCodeFinding(
            relpath=finding.relpath,
            lineno=finding.lineno,
            qualname=finding.name,
            detector="empirical_table",
            tier=tier,
            summary="empirical/generated numeric table literal",
        )
        for finding in empirical_lint.scan_source(text, relpath)
    ]


def _is_spec_builder_name(name: str) -> bool:
    return (
        name.startswith("build_")
        and (
            name.endswith("_run_spec")
            or name.endswith("_training_run_spec")
            or name.endswith("_spec")
        )
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _assignment_value(node: ast.Assign | ast.AnnAssign) -> ast.AST | None:
    if isinstance(node, ast.Assign):
        return node.value
    return node.value


def _is_numeric_constant(node: ast.AST) -> bool:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_NUMERIC_STRING_RE.fullmatch(node.value))
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return True
    return False


def _literal_contains_numeric(node: ast.AST) -> bool:
    if _is_numeric_constant(node):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_literal_contains_numeric(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return any(_literal_contains_numeric(value) for value in node.values)
    return False


def _is_literal_tree(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_numeric_constant(node)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal_tree(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal_tree(key) and _is_literal_tree(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False


def _is_default_bundle_dict(node: ast.Dict) -> bool:
    if len(node.keys) < 3:
        return False
    numeric_values = sum(1 for value in node.values if _is_numeric_constant(value))
    if numeric_values < 2:
        return False
    return any(
        isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and _hp_name_matches(key.value)
        for key in node.keys
    )


def _hp_name_matches(name: str) -> bool:
    tokens = tuple(token for token in re.split(r"[^A-Za-z0-9]+", name.upper()) if token)
    if not tokens or any(token in _DIMENSION_NAME_TOKENS for token in tokens):
        return False
    for lexicon_entry in HP_NAME_LEXICON:
        lexicon_tokens = tuple(lexicon_entry.split("_"))
        if _has_token_sequence(tokens, lexicon_tokens):
            return True
    return False


def _has_token_sequence(tokens: Sequence[str], needle: Sequence[str]) -> bool:
    if len(needle) > len(tokens):
        return False
    return any(tokens[index : index + len(needle)] == tuple(needle) for index in range(len(tokens)))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root to scan",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite ci/data_in_code_baseline.json from current findings",
    )
    parser.add_argument(
        "--allow-baseline-growth",
        action="store_true",
        help="permit baseline growth while writing; intended only for initial seeding",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print findings as JSON objects",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for scanning or regenerating the baseline."""

    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.write_baseline:
        keys = write_baseline(repo_root, allow_growth=args.allow_baseline_growth)
        print(f"wrote {len(keys)} keys to {baseline_path(repo_root)}")
        return 0
    findings = scan_tree(repo_root)
    if args.json:
        for finding in findings:
            print(json.dumps(finding.__dict__, sort_keys=True))
    else:
        for finding in findings:
            print(f"{finding.key} line={finding.lineno} tier={finding.tier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
