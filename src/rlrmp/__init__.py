"""RNNs Learn Robust Motor Policies - Experiment Package.

This package contains the experiment-specific analysis modules, training modules,
and configuration files for the RLRMP project.
"""


def register_experiment_package(registry):
    """Register this experiment package with the feedbax plugin registry.

    Args:
        registry: ExperimentRegistry instance to register with
    """
    from feedbax.plugins.discovery import register_package_from_module_info

    register_package_from_module_info(
        registry=registry,
        package_name="rlrmp",
        package_module_name="rlrmp",
        parts=["part1", "part2", "part3"],
        analysis_module_root="modules.analysis",
        training_module_root="modules.training",
        config_resource_root="config",
        figure_routing={
            "spec_dir_template": "results/{experiment}/figures/{topic}",
            "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
            "spec_format": "json",
            "render_format": "html",
            "create_symlink_in_spec_dir": True,
        },
    )
