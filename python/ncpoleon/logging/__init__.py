import logging
from typing import Literal

from ncpoleon._accelerate.logging import reset_handler as _reset_handler
from ncpoleon._accelerate.logging import set_max_level as _set_max_level
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
    if not 0 <= verbosity <= 3:
        raise ValueError(f"verbosity must be between 0 and 3, got {verbosity}")

    if verbosity == 0:
        _logger.setLevel(logging.WARNING)
    elif verbosity == 1:
        _logger.setLevel(logging.INFO)
    elif verbosity == 2:
        _logger.setLevel(logging.DEBUG)
    elif verbosity == 3:
        _logger.setLevel(5)

    # Lower the `log` crate's global max-level gate so hot-loop `trace!`/`debug!` short-circuit
    # before reaching pyo3-log when they are not wanted. Setting the Python logger level alone does
    # not affect this gate. Reset the pyo3-log cache so the new levels take effect immediately.
    _set_max_level(min(verbosity, 3))
    _reset_handler()


# The Rust logger installs with `filter(Trace)`, which pins the global max level to Trace. Establish
# the default (verbosity 0) so hot loops are not taxed by logging before any explicit call.
set_verbosity_level(0)


__all__ = [
    "set_logging_format",
    "set_notebook",
    "set_verbosity_level",
]
