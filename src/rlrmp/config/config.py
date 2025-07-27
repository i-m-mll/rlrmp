from cProfile import label
from importlib import resources
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Optional, TypeVar
import yaml

import jax.tree as jt

from rlrmp.types import TreeNamespace, dict_to_namespace


logger = logging.getLogger(__name__)


CONFIG_DIR_ENV_VAR_NAME = 'RLRMP_CONFIG_DIR'

def _setup_paths(paths_ns: TreeNamespace):
    base_path = Path(paths_ns.base)

    def _setup_path(path_str: str):
        if path_str == 'base':
            return base_path
        else:
            path = base_path / path_str
            path.mkdir(parents=True, exist_ok=True)
            return path

    return jt.map(_setup_path, paths_ns)


def _normalize_log_level(label: str, lvl: str | int) -> int:
    if isinstance(lvl, str):
        lvl = lvl.strip().upper()
        # this returns an int for a valid name, or the same object if invalid
        try:
            lvl = logging.getLevelNamesMapping()[lvl]
        except KeyError:
            raise ValueError(f"Invalid {label} specified in YAML config: {lvl!r}")
    if not isinstance(lvl, int):
        raise ValueError(f"Cannot parse log level {lvl!r}")
    return lvl


def _setup_logging(logging_ns: TreeNamespace):
    for label in ['file_level', 'console_level', 'pkg_console_levels', 'pkgs_own_files']:
        tree = getattr(logging_ns, label, None)
        if tree is None:
            continue
        tree_normalized = jt.map(
            lambda x: _normalize_log_level(label, x),
            tree,
        )
        setattr(logging_ns, label, tree_normalized)

    return logging_ns


T = TypeVar('T', bound=SimpleNamespace)


def get_user_config_dir():
    """Get user config directory from environment variable, or return None."""
    env_config_dir = os.environ.get(CONFIG_DIR_ENV_VAR_NAME)
    if env_config_dir is None:
        return 
    else:
        return Path(env_config_dir).expanduser() 


def load_config(name: str, config_type: Optional[Literal['training', 'analysis']] = None):
    """Load the contents of a project YAML config file resource as a nested dict."""
    name_parts = name.split('.')
    config_name = name_parts[-1]
    
    # If the user has specified a config directory, try to load the paths config from it
    user_config_dir = get_user_config_dir()
    if user_config_dir is not None:
        subpath = '/'.join(name_parts[:-1])
        try:
            with open(user_config_dir / subpath / f'{name}.yml') as f:
                return yaml.safe_load(f)
        except:  
            logger.info(f'Config file {f"{subpath}/{config_name}.yml"} not found in user config directory; using default.')
    
    if config_type is None:
        subpackage_name = 'rlrmp.config'
    else:
        subpackage_name = f'rlrmp.config.{config_type}'
    
    subpackage_name = '.'.join([subpackage_name, *name_parts[:-1]])
    
    # Otherwise, load the default
    with resources.open_text(subpackage_name, f'{config_name}.yml') as f:
        return yaml.safe_load(f)


def load_config_as_ns(
    name: str, 
    config_type: Optional[Literal['training', 'analysis']] = None,
    to_type: type[T] = TreeNamespace,
) -> T:
    """Load the contents of a project YAML config file resource as a namespace."""
    return dict_to_namespace(load_config(name, config_type), to_type=to_type)


# Load project-wide configuration from YAML resources in the `config` subpackage
CONSTANTS: TreeNamespace = load_config_as_ns("constants")
LOGGING: TreeNamespace = _setup_logging(load_config_as_ns("logging"))
PATHS: TreeNamespace = _setup_paths(load_config_as_ns("paths"))
PLOTLY_CONFIG: TreeNamespace = load_config_as_ns("plotly")
PRNG_CONFIG: TreeNamespace = load_config_as_ns("prng")
STRINGS: TreeNamespace = load_config_as_ns("strings")