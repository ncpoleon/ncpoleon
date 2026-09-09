"""Type stubs for ``picos.valuable``, PICOS 2.6.2.

PICOS ships no ``py.typed``, so everything here was recovered from the
installed package: the class hierarchy from ``__mro__``, signatures from
``inspect.signature``, and return types by building and solving a problem and
inspecting what came back.
"""

from abc import ABC
from typing import Any

import numpy.typing as npt

class Valuable(ABC):
    # PICOS hands the value back in whatever shape fits: a `float` or `complex`
    # for a scalar, a `cvxopt.base.matrix` / `cvxopt.base.spmatrix` for anything
    # larger, and `None` when the object is not valued. CVXOPT ships no type
    # information and the three cases cannot be told apart statically, so this is
    # deliberately `Any` rather than a union that would be wrong either way.
    @property
    def value(self) -> Any: ...
    @value.setter
    def value(self, value: Any) -> None: ...
    @property
    def value_as_matrix(self) -> Any: ...
    @property
    def valued(self) -> bool: ...
    @property
    def np(self) -> Any: ...
    @property
    # `np` below shadows the module inside this class body, so the array type
    # is spelled through `numpy.typing` only.
    def np2d(self) -> npt.NDArray[Any] | None: ...
    @property
    def sp(self) -> Any: ...

class NotValued(RuntimeError, AttributeError): ...
