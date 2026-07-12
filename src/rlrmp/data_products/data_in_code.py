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
    ("argv_rows", "src"): "enforced",
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
    "ADAM_LR",
    "ALPHA",
    "ALPHAS",
    "AMPLITUDE",
    "AMPLITUDES",
    "ANCHOR",
    "ARGS",
    "BATCH",
    "BATCHES",
    "BETA",
    "BRACKETS",
    "BUDGET",
    "CENTER",
    "CHECKPOINT",
    "CLIP",
    "CONDITION",
    "CONDITIONS",
    "COUNT",
    "COVARIANCE",
    "DAMPING",
    "DECAY",
    "DEFAULTS",
    "DELAY",
    "DIGITS",
    "DIRECTION",
    "DIRECTIONS",
    "DT",
    "DURATION",
    "D_REF",
    "EPSILON",
    "ETA",
    "FILES",
    "FRACTION",
    "GAMMA",
    "HIDDEN",
    "HIDDEN_SIZE",
    "HORIZON",
    "INDICES",
    "INIT_POS",
    "INTERPOLATION",
    "INTERVAL",
    "ITERATIONS",
    "KWARGS",
    "LAMBDA",
    "LEARNING_RATE",
    "LEVEL",
    "LEVELS",
    "LR",
    "LRS",
    "MASS",
    "MAX_ITER",
    "NOISE",
    "N_BATCHES",
    "N_REPLICATES",
    "N_ROLLOUT",
    "N_STEPS",
    "N_TRIALS",
    "PARAMS",
    "PERCENTILE",
    "PERT",
    "POS",
    "POSITION",
    "PROBABILITY",
    "PROBES",
    "PROP",
    "RADIUS",
    "REACH",
    "REPLICATE_INDEX",
    "RESTART",
    "RESTARTS",
    "ROWS",
    "RUNS",
    "SAMPLE",
    "SAMPLES",
    "SCALE",
    "SEED",
    "SISU",
    "SLICES",
    "SPECS",
    "STD",
    "STEP",
    "STEPS",
    "SUBWEIGHT",
    "SWEEP",
    "TARGET_POS",
    "TAU",
    "TIME",
    "TIMEOUT",
    "TIMING",
    "TRIALS",
    "VALUES",
    "VEL",
    "VELOCITY",
    "WARMUP",
    "WEIGHT",
    "WEIGHTS",
    "WIDTH",
    "WINDOW",
)

_NUMERIC_STRING_RE = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")
_DIMENSION_NAME_TOKENS = frozenset({"DIM", "DIMS", "DIMENSION", "DIMENSIONS", "SHAPE"})
_NON_PARAMETER_NAME_TOKENS = frozenset(
    {
        "ARTIFACT",
        "CARDINALITY",
        "FILENAME",
        "ISSUE",
        "KIND",
        "LABEL",
        "LOGICAL",
        "PATH",
        "PRODUCER",
        "RELPATH",
        "ROLE",
        "SCHEMA",
        "TOKEN",
    }
)
_TOLERANCE_NAME_TOKENS = frozenset({"ATOL", "RTOL", "TOL", "TOLERANCE"})
_SCHEMA_REFERENCE_SUFFIXES = ("_PARAMS_REF",)
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


_CD137D8_CONFIG_TIER_ALLOWLIST = (
    "src/rlrmp/train/config_materialization.py::CS_REGULARIZED_NN_HIDDEN::hp_constant",
    "src/rlrmp/train/config_materialization.py::CS_STAGE_COUNT::hp_constant",
    "src/rlrmp/train/config_materialization.py::build_hps::default_bundle",
    "src/rlrmp/train/minimax_native/method.py::_build_hps_from_config::default_bundle",
    "src/rlrmp/train/standard.py::_LOSS_WEIGHT_OVERRIDES::hp_constant",
    "src/rlrmp/train/standard.py::_base_hps::default_bundle",
    "src/rlrmp/train/standard.py::_loss_cfg::default_bundle",
    "src/rlrmp/train/training_configs.py::AMPLITUDE_LEVELS::hp_constant",
    "src/rlrmp/train/training_configs.py::BROAD_EPSILON_REFERENCE_REACH_M::hp_constant",
    "src/rlrmp/train/training_configs.py::CS_CONTROL_SCALE::hp_constant",
    "src/rlrmp/train/training_configs.py::CS_POSITION_SCALE::hp_constant",
    "src/rlrmp/train/training_configs.py::CS_VELOCITY_SCALE::hp_constant",
    "src/rlrmp/train/training_configs.py::DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M::hp_constant",
    "src/rlrmp/train/training_configs.py::DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG::hp_constant",
    "src/rlrmp/train/training_configs.py::DEFAULT_PGD_SISU_EXACT_ZERO_MASS::hp_constant",
    "src/rlrmp/train/training_configs.py::DEFAULT_PGD_SISU_LEVELS::hp_constant",
    "src/rlrmp/train/training_configs.py::DEFAULT_SEEN_TARGET_AMPLITUDES_M::hp_constant",
    "src/rlrmp/train/training_configs.py::DEFAULT_SEEN_TARGET_DIRECTIONS_DEG::hp_constant",
    "src/rlrmp/train/training_configs.py::HISTORICAL_020A65B_PGD_RADIUS_15CM::hp_constant",
    "src/rlrmp/train/training_configs.py::ORIGINAL_TARGET_ANCHOR_M::hp_constant",
    "src/rlrmp/train/training_configs.py::PGD_SISU_MAX_RADIUS_SOURCES::hp_constant",
    "src/rlrmp/train/training_configs.py::RAW_STRONG_GAMMA_1P05_RADIUS_15CM::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_BAND16_HELD_OUT_DIRECTIONS::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_BAND36_HELD_OUT_DIRECTIONS::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_BAND8_HELD_OUT_DIRECTIONS::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_BAND_CENTERS_DEG::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_CONST_REACH_M::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_DENSE_N_DIRECTIONS::hp_constant",
    "src/rlrmp/train/training_configs.py::TARGET_SUPPORT_SPARSE_N_DIRECTIONS::hp_constant",
)

_OWNING_SCHEMA_DEFAULT_ALLOWLIST = (
    "src/rlrmp/analysis/math/adversary_equivalence.py::OpenLoopOptimizationConfig::default_bundle",
    "src/rlrmp/analysis/math/cs_released_simulation.py::CSReleasedStochasticNoiseConfig::default_bundle",
    "src/rlrmp/analysis/math/hinf_riccati.py::CostSpec::default_bundle",
    "src/rlrmp/analysis/math/linear_equivalence_certificate.py::CertificateConfig::default_bundle",
    "src/rlrmp/analysis/math/linear_round_trip.py::LinearOptimizationConfig::default_bundle",
    "src/rlrmp/analysis/math/linear_round_trip.py::TeacherFitConfig::default_bundle",
    "src/rlrmp/analysis/math/output_feedback.py::OutputFeedbackConfig::default_bundle",
    "src/rlrmp/eval/steady_state.py::SteadyStatePerturbationBankConfig::default_bundle",
    "src/rlrmp/analysis/robustness_margin.py::RobustnessMarginParams::default_bundle",
    "src/rlrmp/cloud/modal_runner.py::NominalGruRunConfig::default_bundle",
    "src/rlrmp/eval/checkpoint_selection.py::DelayedReachEvalBankSpec::default_bundle",
    "src/rlrmp/eval/output_feedback_rollout_recovery.py::EigenspectrumCoverageConfig::default_bundle",
    "src/rlrmp/eval/output_feedback_rollout_recovery.py::ObserverErrorCoverageConfig::default_bundle",
    "src/rlrmp/eval/output_feedback_rollout_recovery.py::RolloutRecoveryCondition::default_bundle",
    "src/rlrmp/model/stochastic_runtime.py::StochasticRuntimeConfig::default_bundle",
    "src/rlrmp/train/distillation_native/closed_loop_kernel.py::ClosedLoopLossWeights::default_bundle",
    "src/rlrmp/train/distillation_native/losses.py::CSH0DistillationConfig::default_bundle",
    "src/rlrmp/train/distillation_native/losses.py::DistillationLossWeights::default_bundle",
    "src/rlrmp/train/executor/checkpoints.py::AdaptiveEpsilonState::default_bundle",
    "src/rlrmp/train/training_configs.py::BroadFullStateEpsilonTrainingConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::ClosedLoopDistillationConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::CsNominalGruConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::FixedTargetPerturbationTrainingConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::GuidedDistillationConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::MinimaxConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::PgdFullStateEpsilonTrainingConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::PolicyFullStateEpsilonTrainingConfig::default_bundle",
    "src/rlrmp/train/training_configs.py::TargetRelativeMultiTargetTrainingConfig::default_bundle",
)

_PURPOSE_CONSTANT_ALLOWLIST = (
    "src/rlrmp/analysis/sisu_spectrum.py::LOW_SISU_ENDPOINT_REACH_THRESHOLD_M::hp_constant",
    "src/rlrmp/analysis/sisu_spectrum.py::LOW_SISU_PEAK_SPEED_THRESHOLD_M_S::hp_constant",
)


DATA_IN_CODE_ALLOWLIST: dict[str, str] = {
    **_empirical_allowlist(),
    **{
        key: (
            "Canonical config or config-materialization surface established by issue cd137d8; "
            "the values are named training conventions owned by the unified typed schema."
        )
        for key in _CD137D8_CONFIG_TIER_ALLOWLIST
    },
    **{
        key: (
            "The typed config or params class is already the owning schema-default surface; "
            "the detector reports its field defaults because the schema is colocated with its consumer."
        )
        for key in _OWNING_SCHEMA_DEFAULT_ALLOWLIST
    },
    **{
        key: (
            "This value is a diagnostic classification threshold, not a run or evaluation "
            "parameter; the audited inventory independently classified it as a legitimate constant."
        )
        for key in _PURPOSE_CONSTANT_ALLOWLIST
    },
    ("src/rlrmp/cloud/modal_runner.py::DEFAULT_TRAIN_TIMEOUT_SECONDS::hp_constant"): (
        "Operational cloud job timeout rather than a scientific run parameter; it remains a "
        "runner safety bound and is intentionally separate from the governed training spec."
    ),
}


def default_spec_constructor_names(repo_root: Path | None = None) -> frozenset[str]:
    """Return the spec-constructor vocabulary used by the ``spec_flow`` detector."""

    names = set(_SPEC_CONSTRUCTOR_SEEDS)
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    for relpath in (
        "src/rlrmp/runtime/training_run_specs.py",
        "src/rlrmp/train/minimax_native/method.py",
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
            findings.extend(scan_source(text, relpath, constructor_names=constructor_names))
        except SyntaxError as error:
            raise DataInCodePolicyError(
                f"cannot scan syntactically invalid Python source {relpath}: {error}"
            ) from error
    return sorted(findings, key=lambda finding: (finding.key, finding.lineno))


def violations(repo_root: Path) -> list[DataInCodeFinding]:
    """Return live-tree findings that are neither allowlisted nor baselined."""

    baseline = set(load_baseline(repo_root))
    return [
        finding
        for finding in scan_tree(repo_root)
        if finding.key not in DATA_IN_CODE_ALLOWLIST
        and (
            policy_for_finding(finding) == "enforced"
            or (policy_for_finding(finding) == "ratchet" and finding.key not in baseline)
        )
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

    enforced_unallowlisted = {
        key
        for key, finding in findings_by_key.items()
        if policy_for_finding(finding) == "enforced" and key not in allowlist_keys
    }
    if enforced_unallowlisted:
        raise DataInCodePolicyError(
            "enforced data-in-code findings: " + ", ".join(sorted(enforced_unallowlisted))
        )

    current_ratchet = {
        key
        for key, finding in findings_by_key.items()
        if policy_for_finding(finding) == "ratchet" and key not in allowlist_keys
    }
    added = sorted(current_ratchet - baseline_keys)
    stale = sorted(baseline_keys - current_ratchet)
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
    """Write current non-allowlisted ratchet findings as a shrink-only baseline."""

    current = sorted(
        finding.key
        for finding in scan_tree(repo_root)
        if policy_for_finding(finding) == "ratchet" and finding.key not in DATA_IN_CODE_ALLOWLIST
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
        if any(
            keyword.arg is not None and _literal_contains_numeric(keyword.value)
            for keyword in node.keywords
        ):
            self._emit(
                detector="spec_flow",
                lineno=node.lineno,
                summary=f"{name} keyword literal carries run/eval/analysis parameters",
            )

    def _scan_default_bundle_function(self, node: ast.FunctionDef) -> None:
        if not _is_default_bundle_function_name(node.name):
            return
        local_values: dict[str, ast.AST] = {}
        for child in _function_body_nodes(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        local_values[target.id] = child.value
            elif (
                isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
                and child.value is not None
            ):
                local_values[child.target.id] = child.value
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                _record_local_dict_update(child.value, local_values)
            elif isinstance(child, ast.Return):
                if child.value is not None and _returns_default_bundle(
                    child.value,
                    local_values,
                ):
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
        bundle_class = node.name.endswith(("Config", "Params", "Condition", "Weights"))
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
            if (
                (_hp_name_matches(name) or bundle_class)
                and not _name_is_exempt(name)
                and _is_static_parameter_default(value)
            ):
                numeric_hp_defaults += 1
        if numeric_hp_defaults >= 2:
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
        if value is None or not _is_literal_tree(value) or not _literal_contains_numeric(value):
            return
        for name in names:
            if self._scope_kinds and self._scope_kinds[-1] == "class" and not name.isupper():
                continue
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
    return name.startswith("build_") and (
        name.endswith("_run_spec") or name.endswith("_training_run_spec") or name.endswith("_spec")
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
    if isinstance(node, ast.BinOp):
        return _literal_contains_numeric(node.left) or _literal_contains_numeric(node.right)
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in {"array", "asarray", "dict", "list", "set", "tuple"}:
            return any(_literal_contains_numeric(argument) for argument in node.args) or any(
                _literal_contains_numeric(keyword.value) for keyword in node.keywords
            )
        if name in {"field", "Field"}:
            return any(
                keyword.arg in {"default", "default_factory"}
                and _literal_contains_numeric(keyword.value)
                for keyword in node.keywords
            )
        return any(_literal_contains_numeric(argument) for argument in node.args) or any(
            _literal_contains_numeric(keyword.value) for keyword in node.keywords
        )
    if isinstance(node, ast.Lambda):
        return _literal_contains_numeric(node.body)
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
    if isinstance(node, ast.BinOp):
        return _is_literal_tree(node.left) and _is_literal_tree(node.right)
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in {"array", "asarray", "dict", "list", "set", "tuple"}:
            return all(_is_literal_tree(argument) for argument in node.args) and all(
                keyword.arg is not None and _is_literal_tree(keyword.value)
                for keyword in node.keywords
                if keyword.arg != "dtype"
            )
    return False


def _is_default_bundle_dict(node: ast.Dict) -> bool:
    string_keys = list(_dict_string_keys(node))
    if len(string_keys) < 3:
        return False
    numeric_values = sum(1 for child in ast.walk(node) if _is_numeric_constant(child))
    if numeric_values < 2:
        return False
    return any(_hp_name_matches(key) for key in string_keys)


def _dict_string_keys(node: ast.Dict) -> Iterable[str]:
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            yield key.value
        if isinstance(value, ast.Dict):
            yield from _dict_string_keys(value)


def _is_default_bundle_function_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "args_namespace",
            "base_hps",
            "build_hps",
            "config",
            "defaults",
            "loss_cfg",
            "params",
        )
    )


def _function_body_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[ast.AST]:
    stack = list(reversed(node.body))
    while stack:
        child = stack.pop()
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(child))))


def _record_local_dict_update(call: ast.Call, local_values: dict[str, ast.AST]) -> None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "update":
        return
    if not isinstance(call.func.value, ast.Name):
        return
    target = call.func.value.id
    current = local_values.get(target)
    if not isinstance(current, ast.Dict):
        return

    keys = list(current.keys)
    values = list(current.values)
    for argument in call.args:
        if not isinstance(argument, ast.Dict):
            return
        keys.extend(argument.keys)
        values.extend(argument.values)
    for keyword in call.keywords:
        if keyword.arg is None:
            return
        keys.append(ast.Constant(keyword.arg))
        values.append(keyword.value)
    local_values[target] = ast.Dict(keys=keys, values=values)


def _returns_default_bundle(
    node: ast.AST,
    local_values: dict[str, ast.AST],
    *,
    seen: frozenset[str] = frozenset(),
) -> bool:
    if isinstance(node, ast.Name):
        if node.id in seen or node.id not in local_values:
            return False
        return _returns_default_bundle(
            local_values[node.id],
            local_values,
            seen=seen | {node.id},
        )
    if isinstance(node, ast.Dict):
        return _is_default_bundle_dict(node)
    if not isinstance(node, ast.Call):
        return False

    for keyword in node.keywords:
        if keyword.arg is None and _returns_default_bundle(
            keyword.value,
            local_values,
            seen=seen,
        ):
            return True
    for argument in node.args:
        if _returns_default_bundle(argument, local_values, seen=seen):
            return True

    name = _call_name(node.func)
    if name not in {"Namespace", "SimpleNamespace", "TreeNamespace", "dict"}:
        return False
    named_keywords = [keyword for keyword in node.keywords if keyword.arg is not None]
    if len(named_keywords) < 3:
        return False
    numeric_values = sum(
        1 for keyword in named_keywords if _literal_contains_numeric(keyword.value)
    )
    return numeric_values >= 2 and any(
        _hp_name_matches(keyword.arg or "") for keyword in named_keywords
    )


def _is_static_parameter_default(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    if _literal_contains_numeric(node):
        return True
    if isinstance(node, ast.Call) and _call_name(node.func) in {"field", "Field"}:
        return any(
            keyword.arg in {"default", "default_factory"}
            and (
                (
                    isinstance(keyword.value, ast.Name)
                    and keyword.value.id not in {"dict", "list", "set", "tuple"}
                )
                or _literal_contains_numeric(keyword.value)
            )
            for keyword in node.keywords
        )
    return False


def _hp_name_matches(name: str) -> bool:
    if name.upper().endswith(_SCHEMA_REFERENCE_SUFFIXES):
        return False
    tokens = tuple(token for token in re.split(r"[^A-Za-z0-9]+", name.upper()) if token)
    if not tokens or _name_tokens_are_exempt(tokens):
        return False
    for lexicon_entry in HP_NAME_LEXICON:
        lexicon_tokens = tuple(lexicon_entry.split("_"))
        if _has_token_sequence(tokens, lexicon_tokens):
            return True
    return False


def _name_is_exempt(name: str) -> bool:
    tokens = tuple(token for token in re.split(r"[^A-Za-z0-9]+", name.upper()) if token)
    return not tokens or _name_tokens_are_exempt(tokens)


def _name_tokens_are_exempt(tokens: Sequence[str]) -> bool:
    return bool(
        any(token in _DIMENSION_NAME_TOKENS for token in tokens)
        or any(token in _NON_PARAMETER_NAME_TOKENS for token in tokens)
        or any(token in _TOLERANCE_NAME_TOKENS for token in tokens)
    )


def _has_token_sequence(tokens: Sequence[str], needle: Sequence[str]) -> bool:
    if len(needle) > len(tokens):
        return False
    return any(
        all(
            _token_matches(actual, expected)
            for actual, expected in zip(
                tokens[index : index + len(needle)],
                needle,
                strict=True,
            )
        )
        for index in range(len(tokens) - len(needle) + 1)
    )


def _token_matches(actual: str, expected: str) -> bool:
    if actual == expected or actual.rstrip("0123456789") == expected:
        return True
    if expected == "RADIUS" and actual == "RADII":
        return True
    if actual == f"{expected}S":
        return True
    return expected.endswith("Y") and actual == f"{expected[:-1]}IES"


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
