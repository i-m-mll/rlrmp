"""Modal entrypoint for stochastic C&S-fidelity GRU preparation."""

from __future__ import annotations

from typing import Any

import modal
from feedbax.execution.backends import render_modal_app
from feedbax.execution.container import activate_project_venv, collect_source_provenance

from rlrmp.cloud import modal_runner as runner


def _rendered_image() -> modal.Image:
    bundle = runner.build_launcher_spec_bundle(runner.NominalGruRunConfig(), backend="modal")
    namespace: dict[str, Any] = {"__name__": "_feedbax_rendered_modal_app"}
    exec(render_modal_app(bundle.execution_spec), namespace)
    return namespace["image"]


image = _rendered_image()

volume = modal.Volume.from_name(runner.MODAL_VOLUME_NAME, create_if_missing=True)
app = modal.App(runner.APP_NAME, image=image)


@app.function(
    timeout=runner.DEFAULT_TRAIN_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=1,
    volumes={str(runner.MODAL_VOLUME_MOUNT): volume},
)
def _run_payload(payload: dict[str, Any]) -> int:
    activate_project_venv(runner.REMOTE_VENV_DIR)
    return runner.execute_remote_payload(payload, volume_commit=volume.commit)


@app.local_entrypoint()
def main(*args: str) -> None:
    parser = runner.build_parser()
    parsed = parser.parse_args(list(args))
    runner.require_billable_launch_confirmation(parsed)
    config = runner.make_config(parsed)
    if parsed.command in {"dry-run", "local-smoke"}:
        raise SystemExit(runner.main([*args]))
    payload = {
        "command_kind": parsed.command,
        "config": {**config.__dict__, "extra_args": list(config.extra_args)},
        "source_provenance": collect_source_provenance(runner.REPO_ROOT),
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
    raise SystemExit(runner.main())
