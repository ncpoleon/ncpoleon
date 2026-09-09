from __future__ import annotations

from typing import ClassVar, Literal, Self, TypeAlias, final, overload

from ncpoleon.relaxations import (
    ComplexCoefficientsCommutativeConstraint,
    RealCoefficientsCommutativeConstraint,
)

from .. import Polynomial

class BaseCommutativePolynomialElement:
    """The API shared by every element a polynomial can be built from.

    An operator and a monomial expose exactly the same operations, bar `__len__`, which only a monomial
    has. Declaring them once here also lets a type checker resolve them on the
    `CommutativePolynomialElement` union below, which it cannot do when each member declares its own.
    """

    @overload
    def __add__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: CommutativePolynomialElement) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: Polynomial[CommutativeMonomial, complex]) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __radd__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __radd__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: CommutativePolynomialElement) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: Polynomial[CommutativeMonomial, complex]) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: CommutativePolynomialElement) -> CommutativeMonomial: ...
    @overload
    def __mul__(self, other: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: Polynomial[CommutativeMonomial, complex]) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __eq__(
        self, other: float | CommutativePolynomialElement | Polynomial[CommutativeMonomial, float]
    ) -> RealCoefficientsCommutativeConstraint: ...
    @overload
    def __eq__(  # ty: ignore[invalid-method-override]
        self, other: complex | Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: float | CommutativePolynomialElement | Polynomial[CommutativeMonomial, float]
    ) -> RealCoefficientsCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: complex | Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: float | CommutativePolynomialElement | Polynomial[CommutativeMonomial, float]
    ) -> RealCoefficientsCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: complex | Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __neg__(self) -> RealCoefficientsCommutativePolynomial: ...
    def __pow__(self, power: int) -> CommutativeMonomial: ...
    def __hash__(self) -> int: ...
    def adjoint(self) -> Self: ...
    @property
    def moment_matrix_id(self) -> int: ...

@final
class CommutativeOperator(BaseCommutativePolynomialElement): ...

@final
class CommutativeMonomial(BaseCommutativePolynomialElement):
    def __len__(self) -> int: ...

CommutativePolynomialElement: TypeAlias = CommutativeMonomial | CommutativeOperator

# The two classes below are separate `#[pyclass]`es, one per monomorphisation of the Rust
# `Polynomial<CommutativeMonomial, Scalar>`, so they are spelled out rather than left generic in the
# coefficient type: only that way can the promotion to complex coefficients that mixed arithmetic
# performs be stated exactly. They restate only what `Polynomial` cannot already say for these
# parameters; everything it declares in terms of `MonomialType`, `Scalar` and `Self` is already
# exact here and is not repeated.

@final
class RealCoefficientsCommutativePolynomial(Polynomial[CommutativeMonomial, float]):
    """A polynomial in commutative variables with real coefficients."""

    @property
    def is_real(self) -> Literal[True]: ...
    @overload
    def __add__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: CommutativePolynomialElement) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __add__(self, other: Polynomial[CommutativeMonomial, complex]) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __radd__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __radd__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: CommutativePolynomialElement) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __sub__(self, other: Polynomial[CommutativeMonomial, complex]) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: CommutativePolynomialElement) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __mul__(self, other: Polynomial[CommutativeMonomial, complex]) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: float) -> RealCoefficientsCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    @overload
    def __eq__(
        self, other: float | CommutativePolynomialElement | Polynomial[CommutativeMonomial, float]
    ) -> RealCoefficientsCommutativeConstraint: ...
    @overload
    def __eq__(
        self, other: complex | Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: float | CommutativePolynomialElement | Polynomial[CommutativeMonomial, float]
    ) -> RealCoefficientsCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: complex | Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: float | CommutativePolynomialElement | Polynomial[CommutativeMonomial, float]
    ) -> RealCoefficientsCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: complex | Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    __hash__: ClassVar[None]

@final
class ComplexCoefficientsCommutativePolynomial(Polynomial[CommutativeMonomial, complex]):
    """A polynomial in commutative variables with complex coefficients.

    `is_real` is `False` for every instance, including one whose coefficients all happen to have a
    zero imaginary part: it reports which of the two classes this is, not what the coefficients are.
    """

    @property
    def is_real(self) -> Literal[False]: ...
    def __add__(
        self,
        other: complex
        | CommutativePolynomialElement
        | Polynomial[CommutativeMonomial, float]
        | Polynomial[CommutativeMonomial, complex],
    ) -> ComplexCoefficientsCommutativePolynomial: ...
    def __radd__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    def __sub__(
        self,
        other: complex
        | CommutativePolynomialElement
        | Polynomial[CommutativeMonomial, float]
        | Polynomial[CommutativeMonomial, complex],
    ) -> ComplexCoefficientsCommutativePolynomial: ...
    def __rsub__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    def __mul__(
        self,
        other: complex
        | CommutativePolynomialElement
        | Polynomial[CommutativeMonomial, float]
        | Polynomial[CommutativeMonomial, complex],
    ) -> ComplexCoefficientsCommutativePolynomial: ...
    def __rmul__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    def __truediv__(self, other: complex) -> ComplexCoefficientsCommutativePolynomial: ...
    def __eq__(
        self,
        other: complex
        | CommutativePolynomialElement
        | Polynomial[CommutativeMonomial, float]
        | Polynomial[CommutativeMonomial, complex],
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    def __ge__(
        self,
        other: complex
        | CommutativePolynomialElement
        | Polynomial[CommutativeMonomial, float]
        | Polynomial[CommutativeMonomial, complex],
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    def __le__(
        self,
        other: complex
        | CommutativePolynomialElement
        | Polynomial[CommutativeMonomial, float]
        | Polynomial[CommutativeMonomial, complex],
    ) -> ComplexCoefficientsCommutativeConstraint: ...
    __hash__: ClassVar[None]

@overload
def generate_commutative_variables(
    label: str,
    number: int,
    *,
    moment_matrix_id: int = 0,
    starting_index: int = 0,
    real: bool = False,
    projector: bool = False,
    return_identity: Literal[False] = False,
) -> list[CommutativeOperator]: ...
@overload
def generate_commutative_variables(
    label: str,
    number: int,
    *,
    moment_matrix_id: int = 0,
    starting_index: int = 0,
    real: bool = False,
    projector: bool = False,
    return_identity: Literal[True],
) -> tuple[list[CommutativeOperator], CommutativeMonomial]: ...

__all__ = [
    "CommutativeMonomial",
    "CommutativeOperator",
    "CommutativePolynomialElement",
    "ComplexCoefficientsCommutativePolynomial",
    "RealCoefficientsCommutativePolynomial",
    "generate_commutative_variables",
]
