from typing import Generic, Literal, TypeAlias, overload

__all__ = [
    "get_relaxation",
    "RealCoefficientsCommutativeConstraint",
    "ComplexCoefficientsCommutativeConstraint",
    "RealCoefficientsNonCommutativeConstraint",
    "ComplexCoefficientsNonCommutativeConstraint",
    "RealValuedCommutativeMomentMatrix",
    "ComplexValuedCommutativeMomentMatrix",
    "RealValuedNonCommutativeMomentMatrix",
    "ComplexValuedNonCommutativeMomentMatrix",
    "RealValuedCommutativeSdpRelaxation",
    "ComplexValuedCommutativeSdpRelaxation",
    "RealValuedNonCommutativeSdpRelaxation",
    "ComplexValuedNonCommutativeSdpRelaxation",
]

from ncpoleon._typing import PolynomialElements, Scalar
from ncpoleon.polynomials import Polynomial, RewritingStrategy, VectorSpaceElement
from ncpoleon.polynomials.commutative_polynomials import (
    CommutativeOperator,
    CommutativePolynomialElement,
    ComplexCoefficientsCommutativePolynomial,
    RealCoefficientsCommutativePolynomial,
)
from ncpoleon.polynomials.noncommutative_polynomials import (
    ComplexCoefficientsNonCommutativePolynomial,
    NonCommutativeOperator,
    NonCommutativePolynomialElement,
    RealCoefficientsNonCommutativePolynomial,
)

class Constraint(Generic[PolynomialElements, Scalar]):
    @property
    def is_equality(self) -> bool: ...
    @property
    def is_inequality(self) -> bool: ...
    @property
    def lhs(self) -> Polynomial[PolynomialElements, Scalar] | Scalar: ...
    @property
    def rhs(self) -> Polynomial[PolynomialElements, Scalar] | Scalar: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

RealCoefficientsCommutativeConstraint: TypeAlias = Constraint[CommutativePolynomialElement, float]
ComplexCoefficientsCommutativeConstraint: TypeAlias = Constraint[CommutativePolynomialElement, complex]
RealCoefficientsNonCommutativeConstraint: TypeAlias = Constraint[NonCommutativePolynomialElement, float]
ComplexCoefficientsNonCommutativeConstraint: TypeAlias = Constraint[NonCommutativePolynomialElement, complex]

PositionMatrixRowColDataFormat = tuple[list[int], list[int], list[Scalar]]

class PositionMatrix(dict[tuple[int, int], Scalar]): ...
class PositionMatrixPair(tuple[PositionMatrix[Scalar], PositionMatrix[Scalar] | None]): ...

class MomentMatrix(Generic[PolynomialElements, Scalar]):
    @property
    def data(self) -> dict[PolynomialElements, PositionMatrixPair[Scalar]]: ...
    @property
    def size(self) -> int: ...
    def as_row_col_data_format(
        self,
    ) -> dict[
        PolynomialElements,
        tuple[PositionMatrixRowColDataFormat[float], None]
        | tuple[PositionMatrixRowColDataFormat[Scalar], PositionMatrixRowColDataFormat[Scalar]],
    ]: ...
    def __contains__(self, item: PolynomialElements) -> bool: ...
    def __getitem__(self, key: PolynomialElements) -> Scalar: ...
    def get_canonical(self, monomial: PolynomialElements) -> tuple[PolynomialElements, bool, bool]: ...

RealValuedCommutativeMomentMatrix: TypeAlias = MomentMatrix[CommutativePolynomialElement, float]
ComplexValuedCommutativeMomentMatrix: TypeAlias = MomentMatrix[CommutativePolynomialElement, complex]
RealValuedNonCommutativeMomentMatrix: TypeAlias = MomentMatrix[NonCommutativePolynomialElement, float]
ComplexValuedNonCommutativeMomentMatrix: TypeAlias = MomentMatrix[NonCommutativePolynomialElement, complex]

class BaseSdpRelaxation(Generic[PolynomialElements, Scalar]):
    @property
    def objective(self) -> Polynomial[PolynomialElements, Scalar]: ...
    @property
    def moment_matrices(self) -> dict[int, MomentMatrix[PolynomialElements, Scalar]]: ...
    @property
    def localising_moment_matrices_inequalities(
        self,
    ) -> dict[int, list[MomentMatrix[PolynomialElements, Scalar]]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> dict[int, list[MomentMatrix[PolynomialElements, Scalar]]]: ...
    @property
    def moment_equalities(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], Scalar]]: ...
    @property
    def moment_inequalities(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], float]]: ...
    @property
    def is_real(self) -> bool: ...
    @overload
    def rewrite(self, mon_or_poly: PolynomialElements) -> PolynomialElements: ...
    @overload
    def rewrite(
        self, mon_or_poly: Polynomial[PolynomialElements, Scalar]
    ) -> Polynomial[PolynomialElements, Scalar]: ...
    def rewrite(
        self, mon_or_poly: PolynomialElements | Polynomial[PolynomialElements, Scalar]
    ) -> PolynomialElements | Polynomial[PolynomialElements, Scalar]: ...
    def split_into_real_and_imaginary_parts(
        self, polynomial: Polynomial[PolynomialElements, Scalar]
    ) -> tuple[
        dict[PolynomialElements, tuple[float, float | None]],
        dict[PolynomialElements, tuple[float, float | None]] | None,
    ]: ...
    def change_variables(
        self,
        polynomial: Polynomial[PolynomialElements, Scalar],
        mapping: dict[PolynomialElements, VectorSpaceElement[Scalar]],
    ) -> VectorSpaceElement[Scalar]: ...
    @property
    def generating_sets(self) -> dict[int, list[PolynomialElements]]: ...
    @property
    def equalities(self) -> dict[int, list[Polynomial[PolynomialElements, Scalar]]]: ...
    @property
    def inequalities(self) -> dict[int, list[Polynomial[PolynomialElements, Scalar]]]: ...

class RealValuedCommutativeSdpRelaxation(BaseSdpRelaxation[CommutativePolynomialElement, float]):
    @property
    def is_real(self) -> Literal[True]: ...

class ComplexValuedCommutativeSdpRelaxation(BaseSdpRelaxation[CommutativePolynomialElement, complex]):
    @property
    def is_real(self) -> Literal[False]: ...

class RealValuedNonCommutativeSdpRelaxation(BaseSdpRelaxation[NonCommutativePolynomialElement, float]):
    @property
    def is_real(self) -> Literal[True]: ...

class ComplexValuedNonCommutativeSdpRelaxation(BaseSdpRelaxation[NonCommutativePolynomialElement, complex]):
    @property
    def is_real(self) -> Literal[False]: ...

@overload
def get_relaxation(  # type: ignore[overload-overlap]
    variables: list[CommutativeOperator],
    level: int,
    objective: RealCoefficientsCommutativePolynomial,
    *,
    substitutions: dict[CommutativePolynomialElement, float | CommutativePolynomialElement] | None = None,
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
    substitutions: dict[CommutativePolynomialElement, float | CommutativePolynomialElement] | None = None,
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
    substitutions: dict[NonCommutativePolynomialElement, float | NonCommutativePolynomialElement] | None = None,
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
    substitutions: dict[NonCommutativePolynomialElement, float | NonCommutativePolynomialElement] | None = None,
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
