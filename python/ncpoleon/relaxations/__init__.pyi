from typing import Generic, Literal, TypeAlias, overload

from ncpoleon.polynomials import RewritingStrategy, _Polynomial, _PolynomialElements, _Scalar
from ncpoleon.polynomials.commutative_polynomials import (
    CommutativeOperator,
    ComplexCoefficientsCommutativePolynomial,
    RealCoefficientsCommutativePolynomial,
    _CommutativePolynomialElement,
)
from ncpoleon.polynomials.noncommutative_polynomials import (
    ComplexCoefficientsNonCommutativePolynomial,
    NonCommutativeOperator,
    RealCoefficientsNonCommutativePolynomial,
    _NonCommutativePolynomialElement,
)

class _Constraint(Generic[_PolynomialElements, _Scalar]):
    @property
    def is_equality(self) -> bool: ...
    @property
    def is_inequality(self) -> bool: ...
    @property
    def lhs(self) -> _Polynomial[_PolynomialElements, _Scalar] | _Scalar: ...
    @property
    def rhs(self) -> _Polynomial[_PolynomialElements, _Scalar] | _Scalar: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

RealCoefficientsCommutativeConstraint: TypeAlias = _Constraint[_CommutativePolynomialElement, float]
ComplexCoefficientsCommutativeConstraint: TypeAlias = _Constraint[_CommutativePolynomialElement, complex]
RealCoefficientsNonCommutativeConstraint: TypeAlias = _Constraint[_NonCommutativePolynomialElement, float]
ComplexCoefficientsNonCommutativeConstraint: TypeAlias = _Constraint[_NonCommutativePolynomialElement, complex]

_PositionMatrixRowColDataFormat = tuple[list[int], list[int], list[_Scalar]]

class PositionMatrix(dict[tuple[int, int], _Scalar]): ...
class PositionMatrixPair(tuple[PositionMatrix[_Scalar], PositionMatrix[_Scalar] | None]): ...

class MomentMatrix(Generic[_PolynomialElements, _Scalar]):
    @property
    def data(self) -> dict[_PolynomialElements, PositionMatrixPair[_Scalar]]: ...
    @property
    def size(self) -> int: ...
    def as_row_col_data_format(
        self,
    ) -> dict[
        _PolynomialElements,
        tuple[_PositionMatrixRowColDataFormat[float], None]
        | tuple[_PositionMatrixRowColDataFormat[_Scalar], _PositionMatrixRowColDataFormat[_Scalar]],
    ]: ...

class _BaseSdpRelaxation(Generic[_PolynomialElements, _Scalar]):
    @property
    def objective(self) -> _Polynomial[_PolynomialElements, _Scalar]: ...
    @property
    def moment_matrices(self) -> dict[int, MomentMatrix[_PolynomialElements, _Scalar]]: ...
    @property
    def localising_moment_matrices_inequalities(
        self,
    ) -> dict[int, list[MomentMatrix[_PolynomialElements, _Scalar]]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> dict[int, list[MomentMatrix[_PolynomialElements, _Scalar]]]: ...
    @property
    def moment_equalities(self) -> list[tuple[_Polynomial[_PolynomialElements, _Scalar], _Scalar]]: ...
    @property
    def moment_inequalities(self) -> list[tuple[_Polynomial[_PolynomialElements, _Scalar], float]]: ...
    @property
    def is_real(self) -> bool: ...

class RealValuedCommutativeSdpRelaxation(_BaseSdpRelaxation[_CommutativePolynomialElement, float]):
    @property
    def is_real(self) -> Literal[True]: ...

class ComplexValuedCommutativeSdpRelaxation(_BaseSdpRelaxation[_CommutativePolynomialElement, complex]):
    @property
    def is_real(self) -> Literal[False]: ...

class RealValuedNonCommutativeSdpRelaxation(_BaseSdpRelaxation[_NonCommutativePolynomialElement, float]):
    @property
    def is_real(self) -> Literal[True]: ...

class ComplexValuedNonCommutativeSdpRelaxation(_BaseSdpRelaxation[_NonCommutativePolynomialElement, complex]):
    @property
    def is_real(self) -> Literal[False]: ...

@overload
def get_relaxation(  # type: ignore[overload-overlap]
    variables: list[CommutativeOperator],
    level: int,
    objective: RealCoefficientsCommutativePolynomial,
    *,
    substitutions: dict[_CommutativePolynomialElement, float | _CommutativePolynomialElement] | None = None,
    operator_constraints: list[RealCoefficientsCommutativeConstraint] | None = None,
    moment_constraints: list[RealCoefficientsCommutativeConstraint] | None = None,
    normalization_constraints: list[RealCoefficientsCommutativeConstraint] | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    assume_real: bool = False,
) -> RealValuedCommutativeSdpRelaxation: ...
@overload
def get_relaxation(
    variables: list[CommutativeOperator],
    level: int,
    objective: RealCoefficientsCommutativePolynomial | ComplexCoefficientsCommutativePolynomial,
    *,
    substitutions: dict[_CommutativePolynomialElement, float | _CommutativePolynomialElement] | None = None,
    operator_constraints: list[RealCoefficientsCommutativeConstraint | ComplexCoefficientsCommutativeConstraint]
    | None = None,
    moment_constraints: list[RealCoefficientsCommutativeConstraint | ComplexCoefficientsCommutativeConstraint]
    | None = None,
    normalization_constraints: list[RealCoefficientsCommutativeConstraint | ComplexCoefficientsCommutativeConstraint]
    | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    assume_real: bool = False,
) -> ComplexValuedCommutativeSdpRelaxation: ...
@overload
def get_relaxation(  # type: ignore[overload-overlap]
    variables: list[NonCommutativeOperator],
    level: int,
    objective: RealCoefficientsNonCommutativePolynomial,
    *,
    substitutions: dict[_NonCommutativePolynomialElement, float | _NonCommutativePolynomialElement] | None = None,
    operator_constraints: list[RealCoefficientsNonCommutativeConstraint] | None = None,
    moment_constraints: list[RealCoefficientsNonCommutativeConstraint] | None = None,
    normalization_constraints: list[RealCoefficientsNonCommutativeConstraint] | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    assume_real: bool = False,
) -> RealValuedNonCommutativeSdpRelaxation: ...
@overload
def get_relaxation(
    variables: list[NonCommutativeOperator],
    level: int,
    objective: RealCoefficientsNonCommutativePolynomial | ComplexCoefficientsNonCommutativePolynomial,
    *,
    substitutions: dict[_NonCommutativePolynomialElement, float | _NonCommutativePolynomialElement] | None = None,
    operator_constraints: list[RealCoefficientsNonCommutativeConstraint | ComplexCoefficientsNonCommutativeConstraint]
    | None = None,
    moment_constraints: list[RealCoefficientsNonCommutativeConstraint | ComplexCoefficientsNonCommutativeConstraint]
    | None = None,
    normalization_constraints: list[
        RealCoefficientsNonCommutativeConstraint | ComplexCoefficientsNonCommutativeConstraint
    ]
    | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    assume_real: bool = False,
) -> ComplexValuedNonCommutativeSdpRelaxation: ...
