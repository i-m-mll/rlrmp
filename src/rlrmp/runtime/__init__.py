"""Runtime contracts and service materializers for rlrmp.

This package holds lightweight runtime surfaces that are neither training
algorithms nor analysis pipelines: durable run-spec validation, structured-spec
schema registration, and Feedbax Studio record materialization.

Root-level ``rlrmp.paths`` and ``rlrmp.io`` remain cross-cutting utility modules
because both analysis and training code use them directly.
"""

__all__: list[str] = []
