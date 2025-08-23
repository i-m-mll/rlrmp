"""RNNs Learn Robust Motor Policies - Experiment Package.

This package contains the experiment-specific analysis modules, training modules,
and configuration files for the RLRMP project.
"""

def register_experiment_package(registry):
    """Register this experiment package with the feedbax-experiments registry.
    
    Args:
        registry: ExperimentRegistry instance to register with
    """
    from feedbax_experiments.plugins.discovery import register_package_from_module_info
    
    register_package_from_module_info(
        registry=registry,
        package_name="rlrmp",
        package_module_name="rlrmp",
        parts=["part1", "part2"],
        analysis_module_root="analysis.modules",
        training_module_root="training.modules", 
        config_resource_root="config",
    )