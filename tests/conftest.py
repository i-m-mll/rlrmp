"""Shared pytest policy for RLRMP tests."""

from __future__ import annotations

from pathlib import Path
import os

from feedbax.testing import ContractSuiteHooks


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_MANIFEST_PATH = REPO_ROOT / "ci" / "feedbax-contract-suite.toml"
DEFAULT_JAX_CACHE_DIR = REPO_ROOT / "_artifacts" / "test_cache" / "jax_compilation"


def _configure_jax_compilation_cache() -> None:
    """Enable the persistent JAX compilation cache for the test suite."""

    if os.environ.get("RLRMP_TEST_JAX_CACHE", "1").lower() in {"0", "false", "no"}:
        return

    cache_dir = Path(
        os.environ.get("RLRMP_TEST_JAX_CACHE_DIR")
        or os.environ.get("JAX_COMPILATION_CACHE_DIR")
        or DEFAULT_JAX_CACHE_DIR
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    import jax

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    min_compile_time = float(os.environ.get("RLRMP_TEST_JAX_CACHE_MIN_COMPILE_TIME_SECS", "1.0"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time)


_configure_jax_compilation_cache()

_contract_hooks = ContractSuiteHooks(
    root=REPO_ROOT,
    manifest_path=SUITE_MANIFEST_PATH,
    marker="feedbax_contract",
)
pytest_ignore_collect = _contract_hooks.pytest_ignore_collect
pytest_collection_modifyitems = _contract_hooks.pytest_collection_modifyitems
