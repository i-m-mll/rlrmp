"""Modal entrypoint for nominal C&S-fidelity GRU preparation.

Direct local usage:

    uv run python scripts/modal_cs_nominal_gru.py dry-run
    uv run python scripts/modal_cs_nominal_gru.py local-smoke --timeout-seconds 30

Cloud smoke usage:

    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-smoke --timeout-seconds 60

The full Modal training path exists for a later approved launch:

    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-run --timeout-seconds 86400
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from rlrmp.modal_runner import (
    APP_NAME,
    LOCAL_FEEDBAX_DIR,
    LOCAL_JAX_COOKBOOK_DIR,
    REPO_ROOT,
    build_parser,
    dry_run_payload,
    execute_remote_payload,
    make_config,
)


def _ignore_source(path: str) -> bool:
    parts = Path(path).parts
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "_artifacts",
        "worktrees",
        "manuscript.assets",
        "TODO.assets",
    }
    if any(part in ignored for part in parts):
        return True
    return any(part.endswith(".assets") for part in parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "perl")
    .pip_install("uv")
    .add_local_dir(str(REPO_ROOT), "/workspace/rlrmp", ignore=_ignore_source)
    .add_local_dir(str(LOCAL_FEEDBAX_DIR), "/workspace/feedbax", ignore=_ignore_source)
    .add_local_dir(str(LOCAL_JAX_COOKBOOK_DIR), "/workspace/jax-cookbook", ignore=_ignore_source)
    .workdir("/workspace/rlrmp")
)

app = modal.App(APP_NAME, image=image)


@app.function(timeout=60, min_containers=0, max_containers=1)
def _run_payload(payload: dict[str, Any]) -> int:
    return execute_remote_payload(payload)


@app.local_entrypoint()
def main(*args: str) -> None:
    parser = build_parser()
    parsed = parser.parse_args(list(args))
    config = make_config(parsed)
    if parsed.command == "dry-run":
        print(json.dumps(dry_run_payload(config), indent=2, sort_keys=True))
        return
    if parsed.command == "local-smoke":
        from rlrmp.modal_runner import main as local_main

        raise SystemExit(local_main([*args]))
    payload = {
        "command_kind": parsed.command,
        "config": {
            **config.__dict__,
            "extra_args": list(config.extra_args),
        },
    }
    remote_fn = _run_payload.with_options(
        gpu=config.gpu,
        timeout=config.timeout_seconds,
        max_containers=1,
    )
    exit_code = remote_fn.remote(payload)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    from rlrmp.modal_runner import main as local_main

    raise SystemExit(local_main())
