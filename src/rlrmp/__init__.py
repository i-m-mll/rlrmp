"""RNNs Learn Robust Motor Policies experiment package."""


def register_experiment_package(registry):
    """Register this experiment package with the feedbax plugin registry.

    Args:
        registry: ExperimentRegistry instance to register with
    """
    from feedbax.plugins.discovery import register_package_from_module_info
    from rlrmp.analysis.declarative_materialization import (
        register_declarative_materialization_recipes,
    )
    from rlrmp.analysis.training_diagnostics import register_training_diagnostics_recipes
    from rlrmp.analysis.matrix import register_standard_matrix_recipes
    from rlrmp.spec_migrations import ensure_rlrmp_spec_families

    register_package_from_module_info(
        registry=registry,
        package_name="rlrmp",
        package_module_name="rlrmp",
        parts=[],
        analysis_module_root="analysis",
        training_module_root="train",
        config_resource_root="config",
        figure_routing={
            "spec_dir_template": "results/{experiment}/figures/{topic}",
            "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
            "spec_format": "json",
            "render_format": "html",
            "create_symlink_in_spec_dir": True,
        },
    )
    ensure_rlrmp_spec_families()
    register_standard_matrix_recipes()
    register_declarative_materialization_recipes(replace=True)
    register_training_diagnostics_recipes()
