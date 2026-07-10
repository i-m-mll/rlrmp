"""CLI wrapper for closed-loop extLQG distillation preflight commands."""

from rlrmp.train.distillation_entry import distillation_main


if __name__ == "__main__":
    raise SystemExit(distillation_main("closed_loop_distillation"))
