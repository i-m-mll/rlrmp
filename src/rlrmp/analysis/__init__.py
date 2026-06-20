"""Analysis utilities for rlrmp.

Modules in this package are offline analyses (non-training) that operate on
linearised plants, trained checkpoints, or both.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

# Lazy plugin-recipe registration trigger (issue 462bb31).
#
# rlrmp's feedbax plugin registration defers heavy recipe registration out of
# `register_experiment_package` because that runs mid-`feedbax.__init__`, before
# the feedbax public API exists. Real downstream use always imports some
# `rlrmp.analysis.*` module, which runs this package `__init__` first — and by
# then feedbax has finished initializing — so this is the natural trigger point.
# `ensure_rlrmp_recipes_registered` is idempotent and uses an in-progress
# sentinel to break the re-entrant import cycle it creates (its own heavy
# imports re-enter this module).
from rlrmp import ensure_rlrmp_recipes_registered as _ensure_rlrmp_recipes_registered

_ensure_rlrmp_recipes_registered()

del _ensure_rlrmp_recipes_registered
