from __future__ import annotations

from typing import ClassVar, Literal, Self, TypeAlias, final, overload

from ncpoleon.relaxations import (
    ComplexCoefficientsNonCommutativeConstraint,
    RealCoefficientsNonCommutativeConstraint,
)

from .. import Polynomial

class BaseNonCommutativePolynomialElement:
    """The API shared by every element a polynomial can be built from.

    An operator and a monomial expose exactly the same operations, bar `__len__`, which only a monomial
    has. Declaring them once here also lets a type checker resolve them on the
    `NonCommutativePolynomialElement` union below, which it cannot do when each member declares its own.
    """

    @overload
    def __add__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(self, other: NonCommutativePolynomialElement) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(self, other: Polynomial[NonCommutativeMonomial, float]) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(
        self, other: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __radd__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __radd__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: NonCommutativePolynomialElement) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: Polynomial[NonCommutativeMonomial, float]) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(
        self, other: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: NonCommutativePolynomialElement) -> NonCommutativeMonomial: ...
    @overload
    def __mul__(self, other: Polynomial[NonCommutativeMonomial, float]) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(
        self, other: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __eq__(
        self, other: float | NonCommutativePolynomialElement | Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativeConstraint: ...
    @overload
    def __eq__(  # ty: ignore[invalid-method-override]
        self, other: complex | Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: float | NonCommutativePolynomialElement | Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: complex | Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: float | NonCommutativePolynomialElement | Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: complex | Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __neg__(self) -> RealCoefficientsNonCommutativePolynomial: ...
    def __pow__(self, power: int) -> NonCommutativeMonomial: ...
    def __hash__(self) -> int: ...
    def adjoint(self) -> Self: ...
    @property
    def moment_matrix_id(self) -> int: ...

@final
class NonCommutativeOperator(BaseNonCommutativePolynomialElement): ...

@final
class NonCommutativeMonomial(BaseNonCommutativePolynomialElement):
    def __len__(self) -> int: ...

NonCommutativePolynomialElement: TypeAlias = NonCommutativeMonomial | NonCommutativeOperator

# The two classes below are separate `#[pyclass]`es, one per monomorphisation of the Rust
# `Polynomial<NonCommutativeMonomial, Scalar>`, so they are spelled out rather than left generic in the
# coefficient type: only that way can the promotion to complex coefficients that mixed arithmetic
# performs be stated exactly. They restate only what `Polynomial` cannot already say for these
# parameters; everything it declares in terms of `MonomialType`, `Scalar` and `Self` is already
# exact here and is not repeated.

@final
class RealCoefficientsNonCommutativePolynomial(Polynomial[NonCommutativeMonomial, float]):
    """A polynomial in noncommutative variables with real coefficients."""

    @property
    def is_real(self) -> Literal[True]: ...
    @overload
    def __add__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(self, other: NonCommutativePolynomialElement) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(self, other: Polynomial[NonCommutativeMonomial, float]) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __add__(
        self, other: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __radd__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __radd__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: NonCommutativePolynomialElement) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(self, other: Polynomial[NonCommutativeMonomial, float]) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __sub__(
        self, other: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rsub__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: NonCommutativePolynomialElement) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(self, other: Polynomial[NonCommutativeMonomial, float]) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __mul__(
        self, other: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: float) -> RealCoefficientsNonCommutativePolynomial: ...
    @overload
    def __truediv__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @overload
    def __eq__(
        self, other: float | NonCommutativePolynomialElement | Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativeConstraint: ...
    @overload
    def __eq__(
        self, other: complex | Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: float | NonCommutativePolynomialElement | Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativeConstraint: ...
    @overload
    def __ge__(
        self, other: complex | Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: float | NonCommutativePolynomialElement | Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativeConstraint: ...
    @overload
    def __le__(
        self, other: complex | Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    __hash__: ClassVar[None]

@final
class ComplexCoefficientsNonCommutativePolynomial(Polynomial[NonCommutativeMonomial, complex]):
    """A polynomial in noncommutative variables with complex coefficients.

    `is_real` is `False` for every instance, including one whose coefficients all happen to have a
    zero imaginary part: it reports which of the two classes this is, not what the coefficients are.
    """

    @property
    def is_real(self) -> Literal[False]: ...
    def __add__(
        self,
        other: complex
        | NonCommutativePolynomialElement
        | Polynomial[NonCommutativeMonomial, float]
        | Polynomial[NonCommutativeMonomial, complex],
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __radd__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __sub__(
        self,
        other: complex
        | NonCommutativePolynomialElement
        | Polynomial[NonCommutativeMonomial, float]
        | Polynomial[NonCommutativeMonomial, complex],
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __rsub__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __mul__(
        self,
        other: complex
        | NonCommutativePolynomialElement
        | Polynomial[NonCommutativeMonomial, float]
        | Polynomial[NonCommutativeMonomial, complex],
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __rmul__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __truediv__(self, other: complex) -> ComplexCoefficientsNonCommutativePolynomial: ...
    def __eq__(
        self,
        other: complex
        | NonCommutativePolynomialElement
        | Polynomial[NonCommutativeMonomial, float]
        | Polynomial[NonCommutativeMonomial, complex],
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    def __ge__(
        self,
        other: complex
        | NonCommutativePolynomialElement
        | Polynomial[NonCommutativeMonomial, float]
        | Polynomial[NonCommutativeMonomial, complex],
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    def __le__(
        self,
        other: complex
        | NonCommutativePolynomialElement
        | Polynomial[NonCommutativeMonomial, float]
        | Polynomial[NonCommutativeMonomial, complex],
    ) -> ComplexCoefficientsNonCommutativeConstraint: ...
    __hash__: ClassVar[None]

@overload
def generate_noncommutative_variables(
    label: str,
    number: int,
    *,
    moment_matrix_id: int = 0,
    starting_index: int = 0,
    hermitian: bool = False,
    projector: bool = False,
    return_identity: Literal[False] = False,
) -> list[NonCommutativeOperator]: ...
@overload
def generate_noncommutative_variables(
    label: str,
    number: int,
    *,
    moment_matrix_id: int = 0,
    starting_index: int = 0,
    hermitian: bool = False,
    projector: bool = False,
    return_identity: Literal[True],
) -> tuple[list[NonCommutativeOperator], NonCommutativeMonomial]: ...

__all__ = [
    "ComplexCoefficientsNonCommutativePolynomial",
    "NonCommutativeMonomial",
    "NonCommutativeOperator",
    "NonCommutativePolynomialElement",
    "RealCoefficientsNonCommutativePolynomial",
    "generate_noncommutative_variables",
]
