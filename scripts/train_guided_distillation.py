"""CLI wrapper for guided C&S GRU distillation run-spec commands."""

from rlrmp.train.distillation_entry import distillation_main


if __name__ == "__main__":
    raise SystemExit(distillation_main("guided_distillation"))
