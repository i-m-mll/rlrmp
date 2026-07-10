"""Lane B terminal acceptance gate: end-to-end spec-first training smoke (issue 08bb6d4).

This is the declared terminal acceptance gate for the training-migration lane of the
``64a04e0`` Feedbax-native umbrella. Lane B moved rlrmp's C&S GRU and minimax runners,
their checkpoint/resume custody, post-run parity, and cloud launchers off the legacy
argparse/``run.json``-first flow onto Feedbax's public ``TrainingRunSpec`` contract and
generic executor. The individual migration children (``a1b6118`` -> ``7aeda5a`` ->
``95a3865`` -> ``0efc92d`` -> ``5b571ae``, plus ``b047981`` / ``54b0c2e`` / ``799fcb9`` /
``5cc6c90`` / ``d6b7018``) each have their own unit coverage; this gate proves the
lane's *outcome* holds by construction as one executable, CI-enrolled smoke:

1.  **C&S spec-first path (parts a).** A smoke-size composed C&S ``TrainingRunSpec`` is
    built through ``rlrmp.runtime.training_run_specs`` (the composed payload carries the
    Feedbax ``TrainingRunSpec`` under ``FEEDBAX_TRAINING_RUN_SPEC_KEY``), validates as a
    Feedbax ``TrainingRunSpec``, executes a few real batches through the spec-first
    ``run_full_training`` adapter, and the natively-emitted ``TrainingRunManifest``
    resolves back through ``resolve_run_record``.
2.  **Minimax dry-run (part b).** A validated minimax ``TrainingRunSpec`` is built through
    its spec-first construction path; its phase-program / effective-phase fingerprint is
    validated and a tampered fingerprint is rejected -- no full training is run.
3.  **Cloud plans (part c).** Local + Modal + RunPod ``ExecutionPlan``s are rendered from
    the same C&S spec via ``ExecutionSpec.training_run_spec`` with zero provider contact
    (pure rendering), each deriving the Feedbax generic-executor command.

The manifest is emitted under a per-test temporary repo root (the module ``REPO_ROOT``
is redirected for the duration of the smoke) so the gate never writes into the tracked
``_artifacts`` tree. ``skips count as failures`` in this gate: the family is enrolled
``live`` in ``ci/feedbax-contract-suite.toml`` and ``tests/conftest.py`` +
``test_feedbax_contract_meta.py`` forbid SKIP / non-strict XFAIL under the
``feedbax_contract`` marker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.random as jr
import pytest
from feedbax.contracts.training import TrainingRunSpec

import rlrmp.train.executor.cs_supervised as cs_supervised_executor
from rlrmp.cloud.modal_runner import (
    NominalGruRunConfig,
    build_launcher_spec_bundle,
    dry_run_payload,
    spec_lock_payload,
)
from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.runtime.run_specs import resolve_run_record
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    feedbax_training_run_spec_from_payload,
)
from rlrmp.train.cs_nominal_gru import build_parser, run_full_training, write_run_spec
from rlrmp.train.config_cli import parse_config
from rlrmp.train.minimax_native import (
    MINIMAX_METHOD_REF,
    build_hps as build_minimax_hps,
    build_minimax_training_run_spec,
    validate_minimax_run_spec,
)
from rlrmp.train.task_model import build_task_base
from rlrmp.train.training_configs import MinimaxConfig


pytestmark = pytest.mark.feedbax_contract

GENERIC_EXECUTOR_COMMAND = "python -m feedbax execute-training-run-spec"
LANE_B_ISSUE = "08bb6d4"


def _cs_args(**overrides: object) -> argparse.Namespace:
    """Build a smoke-size C&S GRU CLI namespace."""

    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _smoke_cs_args(*, output_dir: Path, spec_dir: Path, **overrides: object) -> argparse.Namespace:
    return _cs_args(
        issue=LANE_B_ISSUE,
        output_dir=str(output_dir),
        spec_dir=str(spec_dir),
        n_train_batches=2,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        controller_lr=1e-3,
        gradient_clip_norm=5.0,
        lr_warmup_batches=1,
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=0.01,
        checkpoint_interval_batches=1,
        log_step=1,
        disable_progress=True,
        quiet_progress=True,
        **overrides,
    )


def test_cs_composed_spec_validates_as_feedbax_training_run_spec(tmp_path: Path) -> None:
    """Part (a) — the composed C&S run spec validates as a Feedbax ``TrainingRunSpec``."""

    args = _smoke_cs_args(
        output_dir=tmp_path / "bulk",
        spec_dir=tmp_path / "cs_smoke",
        dry_run=True,
    )
    result = write_run_spec(args)

    payload = result["run_spec"]
    assert FEEDBAX_TRAINING_RUN_SPEC_KEY in payload
    # No artifacts are written on the dry compose.
    assert not (tmp_path / "bulk").exists()

    spec = feedbax_training_run_spec_from_payload(payload)
    assert isinstance(spec, TrainingRunSpec)
    assert spec.method_ref.key == "rlrmp/cs_supervised/v1"
    assert spec.graph.inline is not None


def test_cs_spec_first_smoke_emits_resolvable_training_run_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Part (a) — execute a few batches spec-first; the native manifest resolves."""

    repo = tmp_path / "repo"
    repo.mkdir()
    # Redirect the module's repo root so the native TrainingRunManifest is emitted under
    # the temporary repo, never into the tracked ``_artifacts`` tree.
    monkeypatch.setattr(cs_supervised_executor, "REPO_ROOT", repo)

    output_dir = tmp_path / "bulk"
    spec_dir = tmp_path / "cs_smoke"
    args = _smoke_cs_args(
        output_dir=output_dir,
        spec_dir=spec_dir,
        full_train=True,
        resume=True,
        allow_fresh_start=True,
    )

    result = run_full_training(args)

    assert result["completed_batches"] == 2
    run_name = Path(result["run_spec_path"]).stem

    # The native TrainingRunManifest is emitted under the redirected temp repo, proving
    # the smoke never writes into the tracked ``_artifacts`` tree.
    manifest_dir = repo / "_artifacts" / "feedbax_runs" / "manifests" / "training_runs"
    manifests = list(manifest_dir.glob("*.json"))
    assert manifests, "native TrainingRunManifest was not emitted under the temp repo"

    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["training_spec"]["kind"] == "RLRMPRunSpec"

    record = resolve_run_record(LANE_B_ISSUE, run_name, repo_root=repo)
    assert record["schema_id"] == "rlrmp.run_spec"
    assert record["schema_version"] == "rlrmp.run_spec.v2"


def _minimax_payload(tmp_path: Path, argv: list[str]) -> dict:
    parsed = parse_config(
        MinimaxConfig,
        [
            "--n-warmup-batches",
            "1",
            "--n-adversary-batches",
            "0",
            "--batch-size",
            "1",
            "--n-replicates",
            "1",
            "--output-dir",
            str(tmp_path / "_artifacts" / "54b0c2e" / "runs" / "spec"),
            *argv,
        ],
        description="test minimax config",
    )
    assert isinstance(parsed, MinimaxConfig)
    args = parsed
    config = parsed.model_dump(mode="python")
    hps = build_minimax_hps(args)
    graph_bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
        key=jr.PRNGKey(args.seed),
    )
    return build_minimax_training_run_spec(
        config,
        graph_spec=graph_bundle.graph_spec,
        output_dir=Path(args.output_dir),
        spec_dir=tmp_path / "results" / "54b0c2e" / "runs" / "spec",
        feedbax_graph={"graph_spec_path": "graph_spec.json", "manifest_path": "manifest.json"},
    )


def test_minimax_dry_run_spec_validates_phase_program_fingerprint(tmp_path: Path) -> None:
    """Part (b) — validated minimax spec + fingerprint guard, no full training."""

    payload = _minimax_payload(tmp_path, [])

    # Phase-program / effective-phase fingerprint validation on the constructed spec.
    validate_minimax_run_spec(payload, spec_dir=tmp_path)

    spec = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    assert spec.method_ref.key == MINIMAX_METHOD_REF
    assert spec.worker_execution.effective_phase.phase_program.initial_phase == "warmup"

    # Tampering the effective-phase identity must be rejected by the fingerprint guard.
    tampered_payload = dict(payload)
    tampered_spec = dict(payload["feedbax_training_run_spec"])
    tampered_spec["worker_execution"]["effective_phase"]["phase_program"]["metadata"] = {
        "phase_program_identity": "tampered"
    }
    tampered_payload["feedbax_training_run_spec"] = tampered_spec
    with pytest.raises(ValueError, match="effective-phase fingerprint mismatch"):
        validate_minimax_run_spec(tampered_payload, spec_dir=tmp_path)


def test_cloud_plans_render_for_all_backends_without_provider_contact() -> None:
    """Part (c) — local/modal/runpod plans render purely from the spec, no provider I/O."""

    config = NominalGruRunConfig(
        experiment=LANE_B_ISSUE,
        run="lane_b_cloud",
        n_train_batches=1,
        batch_size=2,
        n_replicates=1,
        hidden_size=4,
        gradient_clip_norm=5.0,
        runpod_cloud_type="SECURE",
        runpod_gpu_type_ids=("NVIDIA GeForce RTX 4090",),
    )

    per_backend = {
        backend: spec_lock_payload(build_launcher_spec_bundle(config, backend=backend))
        for backend in ("local", "modal", "runpod")
    }
    for backend, payload in per_backend.items():
        assert payload["backend"] == backend
        assert GENERIC_EXECUTOR_COMMAND in payload["derived_runner_command"]

    # Local rendering carries the normalized provider-neutral payload; runpod carries a
    # rendered (not executed) provider command. Neither contacts a provider.
    local_cloud_payload = per_backend["local"]["cloud_payload"]
    assert local_cloud_payload["provider"] == "none"
    assert "runpodctl_create" not in local_cloud_payload
    assert "pod_request" not in local_cloud_payload
    assert "generated_app" not in local_cloud_payload
    assert local_cloud_payload["readiness"] == []
    assert local_cloud_payload["cells"] == []
    assert per_backend["runpod"]["cloud_payload"]["provider"] == "runpod"
    assert "runpodctl pod create" in per_backend["runpod"]["cloud_payload"]["runpodctl_create"]

    # The aggregate dry-run payload renders all three execution plans from one spec source.
    payload = dry_run_payload(config)
    assert set(payload["execution_plans"]) == {"local", "modal", "runpod"}
    assert payload["execution_plans"]["runpod"]["backend"] == "runpod"
