"""Canonical slot names and schema helpers for native-executor rlrmp methods."""

from __future__ import annotations

from dataclasses import dataclass

from feedbax.contracts.worker import (
    BarrierArtifactSinkSpec,
    CheckpointSlotSpec,
    StateSlotSpec,
)

MODEL = "model"
OPTIMIZER = "optimizer"
PRNG = "prng"
COMPLETED_BATCHES = "completed_batches"
OBJECTIVE = "objective"
TRAIN_LOSS = "train_loss"

CONTROLLER = "controller"
CONTROLLER_OPTIMIZER = "controller_optimizer"
ADVERSARY_POPULATION = "adversary_population"
ADVERSARY_OPTIMIZER = "adversary_optimizer"
RNG = "rng"
TRIAL_BATCH = "trial_batch"
CONTROLLER_LOSS = "controller_loss"
ADVERSARY_LOSS = "adversary_loss"

ADAPTIVE_EPSILON_STATE = "adaptive_epsilon_state"
ZERO_ADVERSARY_GUARD = "zero_adversary_guard"
DAMAGE_METRIC = "damage_metric"
EPSILON_SCALE = "epsilon_scale"
ADVERSARY_POLICY = "adversary_policy"

HISTORY_CHUNK_BYTES = "history_chunk_bytes"
ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES = "adaptive_epsilon_diagnostics_bytes"
POLICY_ADVERSARY_DIAGNOSTICS_BYTES = "policy_adversary_diagnostics_bytes"

CS_SUPERVISED_METHOD_REF = "rlrmp/cs_supervised/v1"
ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF = "rlrmp/adaptive_epsilon_curriculum/v1"
POLICY_ADVERSARY_SUPERVISED_METHOD_REF = "rlrmp/policy_adversary_supervised/v1"


@dataclass(frozen=True)
class SlotSchema:
    """Slot groups consumed by R1-R5 method contracts."""

    family: str
    persistent: tuple[str, ...]
    metrics: tuple[str, ...]
    sinks: tuple[str, ...] = ()
    transient: tuple[str, ...] = ()

    @property
    def barrier_slots(self) -> tuple[str, ...]:
        """Slots captured into checkpoint custody."""
        return self.persistent + self.metrics


CS_SUPERVISED_SCHEMA = SlotSchema(
    family=CS_SUPERVISED_METHOD_REF,
    persistent=(MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES),
    metrics=(TRAIN_LOSS,),
    sinks=(HISTORY_CHUNK_BYTES,),
    transient=(OBJECTIVE,),
)
ADAPTIVE_EPSILON_SCHEMA = SlotSchema(
    family=ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    persistent=(
        MODEL,
        OPTIMIZER,
        PRNG,
        COMPLETED_BATCHES,
        ADAPTIVE_EPSILON_STATE,
        ZERO_ADVERSARY_GUARD,
    ),
    metrics=(TRAIN_LOSS, DAMAGE_METRIC, EPSILON_SCALE),
    sinks=(HISTORY_CHUNK_BYTES, ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES),
    transient=(OBJECTIVE,),
)
POLICY_ADVERSARY_SCHEMA = SlotSchema(
    family=POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    persistent=(
        MODEL,
        OPTIMIZER,
        PRNG,
        COMPLETED_BATCHES,
        ADVERSARY_POLICY,
        ADVERSARY_OPTIMIZER,
    ),
    metrics=(TRAIN_LOSS, ADVERSARY_LOSS),
    sinks=(HISTORY_CHUNK_BYTES, POLICY_ADVERSARY_DIAGNOSTICS_BYTES),
    transient=(OBJECTIVE,),
)
def supervised_state_slots(schema: SlotSchema = CS_SUPERVISED_SCHEMA) -> list[StateSlotSpec]:
    """Return Feedbax state-slot declarations for supervised-style families."""
    slots = [
        StateSlotSpec(name=MODEL, role="model"),
        StateSlotSpec(name=OPTIMIZER, role="optimizer"),
        StateSlotSpec(name=PRNG, role="prng"),
        StateSlotSpec(name=COMPLETED_BATCHES, role="auxiliary"),
    ]
    for metric in schema.metrics:
        slots.append(StateSlotSpec(name=metric, role="metric", required=False))
    for sink in schema.sinks:
        slots.append(
            StateSlotSpec(
                name=sink,
                role="auxiliary",
                required=False,
                metadata={"rlrmp_slot_kind": "barrier_artifact_sink"},
            )
        )
    slots.append(StateSlotSpec(name=OBJECTIVE, role="objective", required=False))
    return slots


def minimax_checkpoint_slot_specs() -> list[CheckpointSlotSpec]:
    """Return minimax checkpoint slots for both native executor barriers."""
    return [
        CheckpointSlotSpec(slot=CONTROLLER, axis="replicate"),
        CheckpointSlotSpec(slot=CONTROLLER_OPTIMIZER, axis="replicate"),
        CheckpointSlotSpec(slot=ADVERSARY_POPULATION, axis="adversary_member"),
        CheckpointSlotSpec(slot=ADVERSARY_OPTIMIZER, axis="adversary_member"),
        CheckpointSlotSpec(slot=RNG),
        CheckpointSlotSpec(slot=COMPLETED_BATCHES),
    ]


def checkpoint_slot_specs(schema: SlotSchema) -> list[CheckpointSlotSpec]:
    """Return custody slots, intentionally excluding sink-only artifact slots."""
    return [
        CheckpointSlotSpec(slot=slot, required=slot not in schema.metrics)
        for slot in schema.barrier_slots
    ]


def artifact_sink_specs(schema: SlotSchema) -> list[BarrierArtifactSinkSpec]:
    """Return per-barrier raw artifact sinks for sink slots."""
    return [
        BarrierArtifactSinkSpec(
            slot=slot,
            role="rlrmp_training_artifact",
            logical_name=slot,
            encoding="raw",
            suffix=".bin",
            required=False,
        )
        for slot in schema.sinks
    ]
