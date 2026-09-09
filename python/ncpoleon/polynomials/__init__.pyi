from __future__ import annotations

from typing import Any, ClassVar, Generic, Self, final, overload

from ncpoleon._typing import MonomialType, Scalar
from ncpoleon.relaxations import Constraint

from .commutative_polynomials import generate_commutative_variables
from .noncommutative_polynomials import generate_noncommutative_variables

# `RewritingStrategy` is a `#[pyclass]` Rust enum: a plain, non-instantiable class whose variants
# are class attributes. It is *not* an `enum.Enum` subclass, so it has no `name`/`value` and cannot
# be instantiated or iterated. Unlike `Realness` and `Canonicality` it is not declared `eq`/`eq_int`
# either, so its variants only ever compare equal to themselves, never to their `int` discriminant.
# The `None` variant exists at runtime but cannot be spelled as an attribute in Python source; it is
# only reachable through `getattr(RewritingStrategy, "None")`, hence its absence below.
@final
class RewritingStrategy:
    Greedy: ClassVar[RewritingStrategy]
    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Polynomial(Generic[MonomialType, Scalar]):
    """The API shared by every polynomial class, mirroring the Rust `Polynomial<MonomialType, Scalar>`.

    There is no such class at runtime: each of the four combinations of (non-)commutativity and
    real/complex coefficients is its own `#[pyclass]`, exactly as it is its own monomorphisation on
    the Rust side. This class carries the part of their API that does not depend on which
    combination it is, so that generic code can be written against `Polynomial[MonomialType, Scalar]`
    the way generic Rust code is written against `Polynomial<MonomialType, Scalar>`.

    Everything whose result *does* depend on the combination (arithmetic promoting real coefficients
    to complex ones, comparisons building a constraint of the matching kind) is declared here only
    as widely as is true for all four, and narrowed to a single class by each of them.
    """

    @property
    def is_real(self) -> bool:
        """Whether the coefficients are real, i.e. whether this is a real-coefficients polynomial."""

    def as_dict(self) -> dict[MonomialType, Scalar]:
        """The polynomial as a monomial-to-coefficient mapping. Operators are normalised to monomials."""

    def by_moment_matrix_id(self) -> dict[int, Self]:
        """Split the polynomial into one polynomial per moment matrix identifier."""

    def change_variables(self, mapping: dict[MonomialType, Any]) -> Any:
        """Evaluate the polynomial under a monomial-to-value mapping.

        The values may be anything supporting `+` and `*`, so the return type is whatever those
        operations produce, which no annotation can pin down.
        """

    def degree(self) -> int: ...
    def adjoint(self) -> Self: ...
    def chop(self, delta: float = ...) -> Self: ...
    def is_zero(self, delta: float | None = None) -> bool: ...
    def __neg__(self) -> Self: ...
    def __pow__(self, power: int) -> Self: ...
    # Arithmetic and the comparisons that build a `Constraint` cannot be stated here in terms of
    # `Scalar`: mixing in a complex scalar or a complex-coefficients polynomial promotes the result,
    # so the coefficient type of the result is not necessarily the `Scalar` of `self`. What is true
    # generically is only that the result is a polynomial (or a constraint) over the same
    # `MonomialType` with one of the two coefficient types, and that is what these declare. Each
    # concrete class narrows every one of them to the single class that actually comes back.
    #
    # Combining a polynomial with one of its own kind is the exception: adding or subtracting two
    # polynomials of the same class, or dividing one by a coefficient of its own `Scalar`, cannot
    # promote anything, so those come back as `Self`. Stating them as overloads ahead of the widened
    # ones is what lets generic code over `Polynomial[MonomialType, Scalar]` stay inside `Scalar`
    # instead of falling out into a union it has no way to narrow back.
    @overload
    def __add__(self, other: Self) -> Self: ...
    @overload
    def __add__(
        self, other: MonomialType | complex | Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]
    ) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    def __radd__(self, other: complex) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(
        self, other: MonomialType | complex | Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]
    ) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    def __rsub__(self, other: complex) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    def __mul__(
        self, other: MonomialType | complex | Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]
    ) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    def __rmul__(self, other: complex) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    @overload
    def __truediv__(self, other: Scalar) -> Self: ...
    @overload
    def __truediv__(self, other: complex) -> Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]: ...
    def __eq__(  # ty: ignore[invalid-method-override]
        self, other: MonomialType | complex | Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]
    ) -> Constraint[MonomialType, float] | Constraint[MonomialType, complex]: ...
    def __ge__(
        self, other: MonomialType | complex | Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]
    ) -> Constraint[MonomialType, float] | Constraint[MonomialType, complex]: ...
    def __le__(
        self, other: MonomialType | complex | Polynomial[MonomialType, float] | Polynomial[MonomialType, complex]
    ) -> Constraint[MonomialType, float] | Constraint[MonomialType, complex]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    # Polynomials are unhashable: the `#[pymethods] __eq__` above sets `__hash__` to `None`.
    __hash__: ClassVar[None]

__all__ = [
    "generate_commutative_variables",
    "generate_noncommutative_variables",
    "RewritingStrategy",
]
