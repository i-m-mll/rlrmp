"""Minimal builder-only tensorboardX shim for legacy template imports."""


class SummaryWriter:
    """No-op stand-in; manifest dumping constructs trainers but does not train."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def add_scalar(self, *args, **kwargs) -> None:
        del args, kwargs

    def close(self) -> None:
        return None
