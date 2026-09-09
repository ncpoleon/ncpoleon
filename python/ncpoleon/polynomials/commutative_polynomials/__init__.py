from typing import TypeAlias

from ncpoleon._accelerate.polynomials.commutative_polynomials._monomials import CommutativeMonomial
from ncpoleon._accelerate.polynomials.commutative_polynomials._operators import (
    CommutativeOperator,
    generate_commutative_variables,
)
from ncpoleon._accelerate.polynomials.commutative_polynomials._polynomials import (
    ComplexCoefficientsCommutativePolynomial,
    RealCoefficientsCommutativePolynomial,
)

CommutativePolynomialElement: TypeAlias = CommutativeMonomial | CommutativeOperator

__all__ = [
    "CommutativeMonomial",
    "CommutativeOperator",
    "CommutativePolynomialElement",
    "ComplexCoefficientsCommutativePolynomial",
    "RealCoefficientsCommutativePolynomial",
    "generate_commutative_variables",
]
