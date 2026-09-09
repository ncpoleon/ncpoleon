from typing import TypeAlias

from ncpoleon._accelerate.polynomials.noncommutative_polynomials._monomials import NonCommutativeMonomial
from ncpoleon._accelerate.polynomials.noncommutative_polynomials._operators import (
    NonCommutativeOperator,
    generate_noncommutative_variables,
)
from ncpoleon._accelerate.polynomials.noncommutative_polynomials._polynomials import (
    ComplexCoefficientsNonCommutativePolynomial,
    RealCoefficientsNonCommutativePolynomial,
)

NonCommutativePolynomialElement: TypeAlias = NonCommutativeMonomial | NonCommutativeOperator

__all__ = [
    "ComplexCoefficientsNonCommutativePolynomial",
    "NonCommutativeMonomial",
    "NonCommutativeOperator",
    "NonCommutativePolynomialElement",
    "RealCoefficientsNonCommutativePolynomial",
    "generate_noncommutative_variables",
]
