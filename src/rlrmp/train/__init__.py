"""Hyperparameter constructors for rlrmp training methods.

Modules in this package construct :class:`~feedbax.config.namespace.TreeNamespace`
hyperparameter trees from CLI argparse Namespaces and build task/model skeletons
for training and checkpoint reload paths. The training loops themselves live in
feedbax (:mod:`feedbax.training.train`) and the training-method-specific drivers
live in ``scripts/`` (see e.g. ``scripts/train_minimax.py`` and
``scripts/train_part2_5.py``).

Sub-modules are named by training method (``minimax``, ``standard``), not by
experiment or phase. See the **Script placement** convention in the project
CLAUDE.md for the underlying convention.

Two top-level helpers are re-exported here for convenience:

- :func:`build_hps_minimax` — minimax adversarial training (warm-start +
  alternating controller/adversary gradient updates).
- :func:`build_hps_standard` — non-adversarial standard backprop with the
  composite-loss-mode menu (``running_cost`` / ``softmin`` / ``combined`` /
  ``default``).

:copyright: Copyright 2023-2026 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from rlrmp.train.minimax import build_hps as build_hps_minimax
from rlrmp.train.standard import build_hps as build_hps_standard
from rlrmp.train.task_model import build_task_base, setup_task_model_pair

__all__ = [
    "build_hps_minimax",
    "build_hps_standard",
    "build_task_base",
    "setup_task_model_pair",
]
