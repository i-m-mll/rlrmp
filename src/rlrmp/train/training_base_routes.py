"""Custody routing for complete canonical training-base documents."""

from __future__ import annotations

from feedbax.contracts.training import TrainingRunSpec


def route_training_base(
    spec: TrainingRunSpec,
    *,
    issue: str,
    row_id: str,
) -> TrainingRunSpec:
    """Return ``spec`` with pairwise-disjoint issue-owned run routes.

    The transform operates on a complete ``TrainingRunSpec`` before content
    hashing.  It does not participate in matrix compilation or execution and
    therefore cannot conceal architecture-specific expanded-payload patches.
    """

    artifact_root = f"_artifacts/{issue}/runs/{row_id}"
    tracked_spec_dir = f"results/{issue}/runs/{row_id}"
    checkpoint_dir = f"{artifact_root}/checkpoints"

    config = dict(spec.method_payload.payload.get("config") or {})
    config.update(
        {
            "output_dir": artifact_root,
            "spec_dir": tracked_spec_dir,
            "issue": issue,
            "resume": False,
            "allow_fresh_start": True,
        }
    )
    checkpoint_policy = dict(spec.method_payload.payload.get("checkpoint_policy") or {})
    checkpoint_policy.update({"artifact_root": artifact_root, "tracked_spec_dir": tracked_spec_dir})
    payload = {
        **spec.method_payload.payload,
        "config": config,
        "checkpoint_policy": checkpoint_policy,
    }
    payload.pop("checkpointing", None)
    payload.pop("lr_continuation_mode", None)
    method_payload = spec.method_payload.model_copy(update={"payload": payload})
    return spec.model_copy(
        update={
            "artifacts": spec.artifacts.model_copy(
                update={
                    "artifact_root": artifact_root,
                    "manifest_root": f"{artifact_root}/manifests",
                    "metadata": {
                        **spec.artifacts.metadata,
                        "tracked_spec_dir": tracked_spec_dir,
                    },
                }
            ),
            "checkpoint_progress": spec.checkpoint_progress.model_copy(
                update={
                    "continuation": None,
                    "metadata": {
                        **spec.checkpoint_progress.metadata,
                        "checkpoint_dir": checkpoint_dir,
                    },
                }
            ),
            "method_payload": method_payload,
            "metadata": {
                **{
                    key: value
                    for key, value in spec.metadata.items()
                    if key
                    not in {
                        "source_checkpoint_root",
                        "source_checkpoint_transaction_id",
                        "lr_continuation_schedule",
                    }
                },
                "issue": issue,
                "row_id": row_id,
                "artifact_root": artifact_root,
                "tracked_spec_dir": tracked_spec_dir,
                "execution_start": "fresh",
            },
        }
    )


__all__ = ["route_training_base"]
