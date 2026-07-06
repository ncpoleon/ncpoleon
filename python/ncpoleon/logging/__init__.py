import logging
from typing import Literal

from ncpoleon._accelerate import reset_handler as _reset_handler

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.NOTSET)


def set_logging_format(format: str):
    logging.getLogger().handlers[0].setFormatter(logging.Formatter(format))
    _reset_handler()


def set_verbosity_level(verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3]):
    if verbosity == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbosity == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif verbosity == 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbosity >= 3:
        logging.getLogger().setLevel(logging.NOTSET)

    _reset_handler()


__all__ = [
    "set_logging_format",
    "set_verbosity_level",
]
