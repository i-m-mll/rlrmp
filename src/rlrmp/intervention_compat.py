"""Declarative intervention binding contracts for minimax adversaries."""

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL


LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET = {
    "role": "component_parameter",
    "source_data_id": "linear_dynamics_adversary_params",
    "target_node_id": PLANT_INTERVENOR_LABEL,
    "target_port": "params_override",
    "task_parameter_label": PLANT_INTERVENOR_LABEL,
    "temporal_support": "trajectory",
}


__all__ = ["LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET"]
