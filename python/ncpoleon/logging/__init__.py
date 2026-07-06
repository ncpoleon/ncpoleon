import logging
from typing import Literal

from ncpoleon._accelerate.logging import reset_handler as _reset_handler
from ncpoleon._accelerate.logging import set_notebook as set_notebook

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"

# Package-scoped logger: the Rust side maps its `ncpoleon::...` targets to
# `ncpoleon.*` Python loggers, so configuring this logger also covers the
# Rust handler without touching the root logger (and thus host apps).
_logger = logging.getLogger("ncpoleon")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(FORMAT))
_logger.addHandler(_handler)


def set_logging_format(format: str):
    _handler.setFormatter(logging.Formatter(format))
    _reset_handler()


def set_verbosity_level(verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3]):
    if verbosity == 0:
        _logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        _logger.setLevel(logging.INFO)
    elif verbosity == 2:
        _logger.setLevel(logging.DEBUG)
    elif verbosity >= 3:
        _logger.setLevel(logging.NOTSET)

    _reset_handler()


__all__ = [
    "set_logging_format",
    "set_notebook",
    "set_verbosity_level",
]
