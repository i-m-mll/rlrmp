"""Build a retry-owned Feedbax custody source from the verified baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.random as jr

from rlrmp.runtime.baseline_custody_repair import repair_baseline_custody_source
from rlrmp.train.cs_nominal_gru import build_parser
from rlrmp.train.executor.cs_supervised import (
    build_cs_supervised_native_initial_slots,
    build_run_spec_execution_context,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--repair-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_parser = build_parser()
    context = build_run_spec_execution_context(
        run_parser.parse_args(["--run-spec", str(args.run_spec)]),
        parser=run_parser,
    )
    initial_slots, runtime = build_cs_supervised_native_initial_slots(
        run_spec=context.run_spec,
        hps=context.hps,
        args=context.args,
        key=jr.PRNGKey(int(context.args.seed)),
    )
    result = repair_baseline_custody_source(
        repair_spec_path=args.repair_spec.resolve(),
        source_run_spec=context.run_spec,
        source_templates=initial_slots,
        legacy_model_template=runtime.component("cs_supervised").pair.model,
        legacy_optimizer_state_template=runtime.component("cs_supervised").optimizer_template,
        target_root=None if args.output_root is None else args.output_root.resolve(),
        repo_root=REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "checkpoint_root": str(result.checkpoint_root),
                "transaction_id": result.transaction_id,
                "completed_training_batches": result.completed_training_batches,
                "program_step": result.program_step,
                "barrier_visit_ordinal": result.barrier_visit_ordinal,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
