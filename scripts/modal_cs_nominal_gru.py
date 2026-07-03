"""Modal entrypoint for stochastic C&S-fidelity GRU preparation.

Direct local usage:

    uv run python scripts/modal_cs_nominal_gru.py dry-run
    uv run python scripts/modal_cs_nominal_gru.py local-smoke --timeout-seconds 30

Cloud smoke usage:

    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-smoke --timeout-seconds 60
    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-packing-smoke --n-workers 2

Planned stochastic C&S GRU runs:

    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-run \
        --run cs_stochastic_gru__no_hidden_penalty
    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-run \
        --run cs_stochastic_gru__hidden_penalty --regularized-fidelity

The full Modal training path exists for a later approved launch:

    uv run modal run scripts/modal_cs_nominal_gru.py -- modal-run --timeout-seconds 86400
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from rlrmp.cloud.modal_runner import (
    APP_NAME,
    DEFAULT_TRAIN_TIMEOUT_SECONDS,
    LOCAL_FEEDBAX_DIR,
    LOCAL_JAX_COOKBOOK_DIR,
    MODAL_VOLUME_MOUNT,
    MODAL_VOLUME_NAME,
    REPO_ROOT,
    activate_project_venv,
    build_parser,
    collect_source_provenance,
    dry_run_payload,
    execute_remote_payload,
    make_config,
    require_billable_launch_confirmation,
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


_PATCH_AND_SYNC = r"""
from pathlib import Path

replacements = {
    "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax": "/workspace/feedbax",
    "../../../20 Feedbax/feedbax": "/workspace/feedbax",
    "/Users/mll/Main/10 Projects/05 Utils/jax-cookbook": "/workspace/jax-cookbook",
    "../../../../05 Utils/jax-cookbook": "/workspace/jax-cookbook",
    "../../../../../05 Utils/jax-cookbook": "/workspace/jax-cookbook",
}
for filename in (
    "/workspace/rlrmp/pyproject.toml",
    "/workspace/rlrmp/uv.lock",
    "/workspace/feedbax/pyproject.toml",
    "/workspace/feedbax/uv.lock",
):
    path = Path(filename)
    if not path.exists():
        continue
    text = path.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)
"""


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "perl")
    .pip_install("uv")
    .env(
        {
            "PYTHONPATH": (
                "/workspace/rlrmp/src:/workspace/feedbax/src:"
                "/workspace/jax-cookbook/src"
            ),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    .add_local_dir(str(REPO_ROOT), "/workspace/rlrmp", copy=True, ignore=_ignore_source)
    .add_local_dir(str(LOCAL_FEEDBAX_DIR), "/workspace/feedbax", copy=True, ignore=_ignore_source)
    .add_local_dir(
        str(LOCAL_JAX_COOKBOOK_DIR),
        "/workspace/jax-cookbook",
        copy=True,
        ignore=_ignore_source,
    )
    .workdir("/workspace/rlrmp")
    .run_commands(
        f"python - <<'PY'\n{_PATCH_AND_SYNC}\nPY",
        "uv sync",
        'uv pip install -U "jax[cuda12]"',
    )
)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(
    timeout=DEFAULT_TRAIN_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=1,
    volumes={str(MODAL_VOLUME_MOUNT): volume},
)
def _run_payload(payload: dict[str, Any]) -> int:
    activate_project_venv()
    return execute_remote_payload(payload, volume_commit=volume.commit)


@app.local_entrypoint()
def main(*args: str) -> None:
    parser = build_parser()
    parsed = parser.parse_args(list(args))
    require_billable_launch_confirmation(parsed)
    config = make_config(parsed)
    if parsed.command == "dry-run":
        print(json.dumps(dry_run_payload(config), indent=2, sort_keys=True))
        return
    if parsed.command == "local-smoke":
        from rlrmp.cloud.modal_runner import main as local_main

        raise SystemExit(local_main([*args]))
    payload = {
        "command_kind": parsed.command,
        "config": {
            **config.__dict__,
            "extra_args": list(config.extra_args),
        },
        "source_provenance": collect_source_provenance(REPO_ROOT),
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
    from rlrmp.cloud.modal_runner import main as local_main

    raise SystemExit(local_main())
