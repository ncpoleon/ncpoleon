from __future__ import annotations

from typing import TypeAlias, TypeVar

import numpy as np

from ncpoleon.polynomials.commutative_polynomials import CommutativeMonomial
from ncpoleon.polynomials.noncommutative_polynomials import NonCommutativeMonomial

# Shared TypeVars used across the polynomials, relaxations, solve and export modules.
# Defining them here (rather than re-declaring identically-named TypeVars in each
# module) gives every module the same TypeVar object

# The parameter every generic class in the package is keyed on, mirroring the `MonomialType` of the
# Rust `Polynomial<MonomialType, Scalar>`, `Constraint<MonomialType, Scalar>` and
# `RustMomentMatrix<Scalar, MonomialType>`: what these actually *store* and *hand back* is always a
# monomial, never an operator. Operators (and the float `1.0`) are only ever accepted in argument
# position, where they are coerced to a monomial, so they widen the parameters of the concrete
# classes rather than the type parameter itself.
MonomialType = TypeVar("MonomialType", CommutativeMonomial, NonCommutativeMonomial)

Scalar = TypeVar("Scalar", float, complex)

RealMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
ComplexMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
RealOrComplexMatrix: TypeAlias = RealMatrix | ComplexMatrix
