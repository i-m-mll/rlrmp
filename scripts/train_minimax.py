"""Generated-config CLI for the sole native minimax executor."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import jax.random as jr
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.model.feedbax_graph import (
    build_runtime_rlrmp_feedbax_graph_bundle,
    write_graph_spec_bundle,
)
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.checkpoint_custody import (
    MINIMAX_ADVERSARIAL_BARRIER,
    MINIMAX_WARMUP_BARRIER,
)
from rlrmp.runtime.jax_config import assert_jax_x64_disabled
from rlrmp.train.config_cli import build_config_parser
from rlrmp.train.minimax_native import (
    build_hps,
    build_minimax_training_run_spec,
    execute_minimax_training_run_spec_native,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
    verify_minimax_checkpoint_resume,
)
from rlrmp.train.run_spec_authoring import derive_spec_dir, derive_spec_path
from rlrmp.train.resume_control import emit_launch_continuation, resolve_launch_continuation
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.training_configs import MinimaxConfig

logger = logging.getLogger(__name__)


def author_minimax_training_run_spec(config: MinimaxConfig) -> TrainingRunSpec:
    """Materialize a typed config into one governed native TrainingRunSpec."""

    output_dir = Path(config.output_dir)
    spec_dir = Path(config.spec_dir) if config.spec_dir is not None else derive_spec_dir(output_dir)
    mkdir_p(output_dir)
    mkdir_p(spec_dir)
    spec_path = derive_spec_path(output_dir)
    mkdir_p(spec_path.parent)
    hps = build_hps(config)
    pair = setup_task_model_pair(hps, key=jr.split(jr.PRNGKey(config.seed), 3)[0])
    graph_bundle = build_runtime_rlrmp_feedbax_graph_bundle(hps, pair.model)
    graph_path = write_graph_spec_bundle(graph_bundle, spec_dir)
    payload = build_minimax_training_run_spec(
        config,
        graph_spec=graph_bundle.graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
        feedbax_graph=graph_bundle.to_run_metadata(graph_spec_path=graph_path.name),
    )
    validate_minimax_run_spec(payload, spec_dir=spec_dir)
    spec_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def training_run_spec_from_argv(argv: list[str]) -> TrainingRunSpec:
    """Resolve either a materialized spec or a generated typed config."""

    if argv[:1] == ["--training-run-spec"]:
        filtered = [arg for arg in argv if arg != "--verify-resume-only"]
        if len(filtered) != 2:
            raise ValueError("--training-run-spec expects exactly one path")
        return minimax_training_run_spec_from_file(filtered[1])
    parser = build_config_parser(MinimaxConfig, description="Run native minimax training.")
    parser.add_argument(
        "--verify-resume-only",
        action="store_true",
        help=(
            "Load the configured checkpoint, run strict binding/integrity/ABI "
            "resume gates, print the continuation summary, and exit without training."
        ),
    )
    values = vars(parser.parse_args(argv))
    values.pop("verify_resume_only")
    config = MinimaxConfig.model_validate(values)
    assert isinstance(config, MinimaxConfig)
    return author_minimax_training_run_spec(config)


def run_training(training_run_spec: TrainingRunSpec | dict[str, Any]) -> Any:
    """Execute a validated spec through the sole native minimax executor."""

    spec = (
        training_run_spec
        if isinstance(training_run_spec, TrainingRunSpec)
        else TrainingRunSpec.model_validate(training_run_spec)
    )
    config = MinimaxConfig.model_validate(minimax_training_run_spec_to_config(spec))
    assert_jax_x64_disabled("minimax training", allow_x64=config.allow_x64)
    checkpoint_root = Path(config.output_dir) / "checkpoints_adversarial"
    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=config.resume,
        allow_fresh_start=config.allow_fresh_start,
        stop_target_batches=config.n_warmup_batches + config.n_adversary_batches,
        completed_batches_from_latest=lambda path: _completed_minimax_batches(path, config),
    )
    emit_launch_continuation(continuation, logger=logger)
    return execute_minimax_training_run_spec_native(
        spec,
        run_id=Path(config.output_dir).name,
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        checkpoint_root=checkpoint_root,
        resume=continuation.resume,
        manifest_conflict_policy="reuse-identical",
    )


def _completed_minimax_batches(path: Path, config: MinimaxConfig) -> int:
    coordinate = json.loads(path.read_text(encoding="utf-8")).get("completed_coordinate", {})
    total = config.n_warmup_batches + config.n_adversary_batches
    if coordinate.get("phase") == "done":
        return total
    if coordinate.get("completed_barrier") == MINIMAX_WARMUP_BARRIER:
        return config.n_warmup_batches
    if coordinate.get("completed_barrier") == MINIMAX_ADVERSARIAL_BARRIER:
        return min(total, config.n_warmup_batches + int(coordinate.get("global_step", 0)))
    raise ValueError(f"unsupported minimax checkpoint coordinate in {path}: {coordinate!r}")


def main(argv: list[str] | None = None) -> int:
    """Run training or the checkpoint-only resume preflight."""

    args = list(sys.argv[1:] if argv is None else argv)
    verify_only = "--verify-resume-only" in args
    spec = training_run_spec_from_argv(args)
    result = verify_minimax_checkpoint_resume(spec) if verify_only else run_training(spec)
    if verify_only:
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
