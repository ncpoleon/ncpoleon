from __future__ import annotations

from typing import TypeVar

from ncpoleon.polynomials.commutative_polynomials import CommutativePolynomialElement
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativePolynomialElement

# Shared TypeVars used across the polynomials, relaxations, solve and export modules.
# Defining them here (rather than re-declaring identically-named TypeVars in each
# module) gives every module the *same* TypeVar object, so type checkers can follow
# generic flow across module boundaries.

PolynomialElements = TypeVar("PolynomialElements", CommutativePolynomialElement, NonCommutativePolynomialElement)
Scalar = TypeVar("Scalar", float, complex)
