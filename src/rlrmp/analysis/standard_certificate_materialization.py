"""Compatibility wrapper for :mod:`rlrmp.analysis.pipelines.standard_certificate_materialization`."""

from importlib import import_module as _import_module
from sys import modules as _modules

_compat_name = __name__
_module = _import_module("rlrmp.analysis.pipelines.standard_certificate_materialization")
globals().update(_module.__dict__)
_modules[_compat_name] = _module
