"""AST data-lint: flag generated empirical data baked as source-code constants.

Generated or adopted empirical datasets (calibration tables, budget anchors)
must live in tracked data products loaded fail-closed by identity, not as
module-level source constants (see issue ea6ccb4). This lint flags module-level
container literals (``dict`` / ``tuple`` / ``list`` / ``set``) whose float leaves
carry high precision **and** whose high-precision-leaf cardinality exceeds a small
threshold, unless the assignment target is allowlisted-with-rationale here.

It is deliberately AST-based (a regex lint does not satisfy the ea6ccb4
criterion): the scanner parses each module and inspects only top-level
assignments, so it distinguishes a baked literal from a value produced by a
loader call (``X = load_...()`` is a ``Call``, not a container literal, and is
never flagged). It ignores:

* scalars (dimension constants like ``BROAD_EPSILON_DIM = 8``, solver tolerances,
  single adopted scalars such as ``R_weight``) -- only multi-entry container
  literals are considered;
* low-precision numeric conventions (amplitude factors, fractions of reach);
* enum-like string tuples / labels (no float leaves);
* everything under ``tests/``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ALLOWLIST",
    "GeneratedConstantFinding",
    "HIGH_PRECISION_SIGFIG_THRESHOLD",
    "MIN_HIGH_PRECISION_CARDINALITY",
    "scan_source",
    "scan_tree",
    "significant_figures",
    "violations",
]

# A float leaf counts as "high precision" when it carries strictly more than this
# many significant figures. 6 sits in the middle of the ea6ccb4 "~4-6 sig figs"
# band and cleanly separates adopted empirical values (15+ sig figs) from
# conventional constants (1-3 sig figs).
HIGH_PRECISION_SIGFIG_THRESHOLD = 6

# A container literal is flagged when it has strictly more than this many
# high-precision float leaves (cardinality > 3).
MIN_HIGH_PRECISION_CARDINALITY = 3

# Allowlist of module-level container literals that carry high-precision float
# tables but are intentionally kept as source constants with explicit provenance.
# Keys are ``"<repo-relative-path>::<name>"``; values are the required rationale.
ALLOWLIST: dict[str, str] = {
    "src/rlrmp/train/cs_perturbation_training.py::PGD_SISU_MAX_RADIUS_SOURCES": (
        "Adopted-run-derived analytical / historical PGD radius sources. Each entry "
        "carries inline source_issue/source_note/source_kind provenance (a7dad8a "
        "strong analytical anchor, c92ebd8 output-feedback rollout budgets, 020a65b "
        "historical replay radius). These are explicitly documented adopted anchors, "
        "not silently baked generated data, so they are allowlisted-with-rationale "
        "rather than migrated to a governed product (ea6ccb4)."
    ),
}


@dataclass(frozen=True)
class GeneratedConstantFinding:
    """A flagged module-level generated-data constant literal."""

    relpath: str
    lineno: int
    name: str
    n_high_precision: int
    n_float_leaves: int

    @property
    def key(self) -> str:
        return f"{self.relpath}::{self.name}"


def significant_figures(value: float) -> int:
    """Return the number of significant figures in ``value``'s shortest repr."""

    text = repr(float(value))
    mantissa = text.split("e")[0].split("E")[0]
    digits = mantissa.lstrip("-").replace(".", "").lstrip("0").rstrip("0")
    return len(digits)


def _float_leaves(node: ast.AST) -> list[float]:
    leaves: list[float] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, float):
            leaves.append(child.value)
        elif (
            isinstance(child, ast.UnaryOp)
            and isinstance(child.op, ast.USub)
            and isinstance(child.operand, ast.Constant)
            and isinstance(child.operand.value, float)
        ):
            leaves.append(-child.operand.value)
    return leaves


def _assignment_targets_and_value(
    node: ast.stmt,
) -> tuple[list[str], ast.expr | None]:
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        return names, node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        if isinstance(node.target, ast.Name):
            return [node.target.id], node.value
    return [], None


def scan_source(text: str, relpath: str) -> list[GeneratedConstantFinding]:
    """Return findings for module-level generated-data constant literals in ``text``."""

    tree = ast.parse(text)
    findings: list[GeneratedConstantFinding] = []
    for node in tree.body:  # module level only
        names, value = _assignment_targets_and_value(node)
        if value is None or not names:
            continue
        if not isinstance(value, (ast.Dict, ast.Tuple, ast.List, ast.Set)):
            continue
        floats = _float_leaves(value)
        high_precision = [
            f for f in floats if significant_figures(f) > HIGH_PRECISION_SIGFIG_THRESHOLD
        ]
        if len(high_precision) <= MIN_HIGH_PRECISION_CARDINALITY:
            continue
        for name in names:
            findings.append(
                GeneratedConstantFinding(
                    relpath=relpath,
                    lineno=node.lineno,
                    name=name,
                    n_high_precision=len(high_precision),
                    n_float_leaves=len(floats),
                )
            )
    return findings


def scan_tree(src_root: Path, *, repo_root: Path) -> list[GeneratedConstantFinding]:
    """Scan every ``*.py`` under ``src_root`` (never under ``tests/``)."""

    findings: list[GeneratedConstantFinding] = []
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relpath = path.resolve().relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            findings.extend(scan_source(text, relpath))
        except SyntaxError:
            continue
    return findings


def violations(
    src_root: Path,
    *,
    repo_root: Path,
) -> list[GeneratedConstantFinding]:
    """Return findings that are not allowlisted-with-rationale."""

    return [
        finding
        for finding in scan_tree(src_root, repo_root=repo_root)
        if finding.key not in ALLOWLIST
    ]
