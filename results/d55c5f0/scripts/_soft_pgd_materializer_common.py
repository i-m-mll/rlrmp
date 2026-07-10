"""Shared constants and guarded c92-loader helpers for d55 soft-PGD materializers."""

from __future__ import annotations
from rlrmp.paths import portable_repo_path

import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType

from rlrmp.paths import REPO_ROOT


ISSUE = "d55c5f0"
C92_ISSUE = "c92ebd8"
C92_SCRIPTS_DIR = REPO_ROOT / "results" / C92_ISSUE / "scripts"


@dataclass(frozen=True)
class SoftRow:
    """One first-batch d55 soft-PGD row."""

    run_id: str
    label: str
    legend: str
    training_key: str
    gamma_factor: float
    color: str


SOFT_ROWS: tuple[SoftRow, ...] = (
    SoftRow(
        run_id="soft_pgd_ofb1p05",
        label="Soft PGD OFB gamma 1.05",
        legend="Soft PGD gamma 1.05 GRU",
        training_key="soft_pgd_ofb1p05",
        gamma_factor=1.05,
        color="#2563eb",
    ),
    SoftRow(
        run_id="soft_pgd_ofb1p4",
        label="Soft PGD OFB gamma 1.4",
        legend="Soft PGD gamma 1.4 GRU",
        training_key="soft_pgd_ofb1p4",
        gamma_factor=1.4,
        color="#7c3aed",
    ),
    SoftRow(
        run_id="soft_pgd_ofb1p8",
        label="Soft PGD OFB gamma 1.8",
        legend="Soft PGD gamma 1.8 GRU",
        training_key="soft_pgd_ofb1p8",
        gamma_factor=1.8,
        color="#be185d",
    ),
)
SOFT_RUN_IDS = tuple(row.run_id for row in SOFT_ROWS)


def assert_soft_inputs_ready() -> None:
    """Fail clearly when d55 run specs or synced bulk artifacts are missing."""

    missing: list[str] = []
    for row in SOFT_ROWS:
        run_spec_path = REPO_ROOT / "results" / ISSUE / "runs" / f"{row.run_id}.json"
        artifact_dir = REPO_ROOT / "_artifacts" / ISSUE / "runs" / row.run_id
        if not run_spec_path.exists():
            missing.append(str(run_spec_path.relative_to(REPO_ROOT)))
        if not artifact_dir.exists():
            missing.append(str(artifact_dir.relative_to(REPO_ROOT)))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Missing d55 soft-PGD post-run inputs. Sync/record completed artifacts "
            "for all three rows before materializing diagnostics:\n"
            f"  - {joined}"
        )


def load_c92_module(module_name: str, filename: str) -> ModuleType:
    """Load one c92 experiment-local helper module by exact file path."""

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    path = C92_SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing c92 helper module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


repo_rel = portable_repo_path
