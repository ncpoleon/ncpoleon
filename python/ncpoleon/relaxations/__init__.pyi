from collections.abc import Mapping
from typing import Any, ClassVar, Generic, Literal, TypeAlias, final, overload

__all__ = [
    "get_relaxation",
    "Realness",
    "Canonicality",
    "Hermiticity",
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

from ncpoleon._typing import MonomialType, Scalar
from ncpoleon.polynomials import Polynomial, RewritingStrategy
from ncpoleon.polynomials.commutative_polynomials import (
    CommutativeMonomial,
    CommutativeOperator,
    CommutativePolynomialElement,
    ComplexCoefficientsCommutativePolynomial,
    RealCoefficientsCommutativePolynomial,
)
from ncpoleon.polynomials.noncommutative_polynomials import (
    ComplexCoefficientsNonCommutativePolynomial,
    NonCommutativeMonomial,
    NonCommutativeOperator,
    NonCommutativePolynomialElement,
    RealCoefficientsNonCommutativePolynomial,
)

# `Constraint`, `MomentMatrix` and `BaseSdpRelaxation` below play the role their Rust counterparts
# do: a single generic definition, `Constraint<MonomialType, Scalar>`,
# `RustMomentMatrix<Scalar, MonomialType>` and `SdpRelaxation<..>`, carrying the whole API. Each is
# then monomorphised into the four `#[pyclass]`es that actually exist at runtime, one per
# combination of (non-)commutativity and real/complex coefficients, and those are declared as plain
# non-generic classes here too. They only restate a member when they can say something the generic
# definition cannot: which concrete polynomial class comes back, or a `Literal` for `is_real`.
# Members returning an invariant container (`list`, `dict`) are deliberately left to the generic
# definition, where `MonomialType` and `Scalar` already pin them down exactly.

class Constraint(Generic[MonomialType, Scalar]):
    """An equality or inequality between two polynomials, or between a polynomial and a scalar."""

    @property
    def is_equality(self) -> bool: ...
    @property
    def is_inequality(self) -> bool: ...
    @property
    def lhs(self) -> Polynomial[MonomialType, Scalar] | Scalar: ...
    @property
    def rhs(self) -> Polynomial[MonomialType, Scalar] | Scalar: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class RealCoefficientsCommutativeConstraint(Constraint[CommutativeMonomial, float]):
    @property
    def lhs(self) -> RealCoefficientsCommutativePolynomial | float: ...
    @property
    def rhs(self) -> RealCoefficientsCommutativePolynomial | float: ...

@final
class ComplexCoefficientsCommutativeConstraint(Constraint[CommutativeMonomial, complex]):
    @property
    def lhs(self) -> ComplexCoefficientsCommutativePolynomial | complex: ...
    @property
    def rhs(self) -> ComplexCoefficientsCommutativePolynomial | complex: ...

@final
class RealCoefficientsNonCommutativeConstraint(Constraint[NonCommutativeMonomial, float]):
    @property
    def lhs(self) -> RealCoefficientsNonCommutativePolynomial | float: ...
    @property
    def rhs(self) -> RealCoefficientsNonCommutativePolynomial | float: ...

@final
class ComplexCoefficientsNonCommutativeConstraint(Constraint[NonCommutativeMonomial, complex]):
    @property
    def lhs(self) -> ComplexCoefficientsNonCommutativePolynomial | complex: ...
    @property
    def rhs(self) -> ComplexCoefficientsNonCommutativePolynomial | complex: ...

PositionMatrixRowColDataFormat: TypeAlias = tuple[list[int], list[int], list[Scalar]]

# `Realness` and `Canonicality` are `#[pyclass(eq, eq_int)]` Rust enums: plain classes whose
# variants are class attributes, comparable to each other and to their `int` discriminant. They are
# not `enum.Enum` subclasses, so they have no `name`/`value` and cannot be instantiated or iterated.
@final
class Realness:
    Real: ClassVar[Realness]
    Complex: ClassVar[Realness]
    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Canonicality:
    Canonical: ClassVar[Canonicality]
    Adjoint: ClassVar[Canonicality]
    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Hermiticity:
    Hermitian: ClassVar[Hermiticity]
    NonHermitian: ClassVar[Hermiticity]
    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class MomentMatrix(Generic[MonomialType, Scalar]):
    @property
    def size(self) -> int: ...
    def as_row_col_data_format(
        self,
    ) -> dict[
        MonomialType,
        tuple[PositionMatrixRowColDataFormat[Scalar], Realness],
    ]: ...
    def __contains__(self, item: MonomialType) -> bool: ...
    def get_canonical(self, monomial: MonomialType) -> tuple[MonomialType, Canonicality, Realness]: ...

# The four concrete moment matrices only widen what they accept: a monomial argument may equally be
# an operator, or the float `1.0`, both of which are coerced to a monomial.

@final
class RealValuedCommutativeMomentMatrix(MomentMatrix[CommutativeMonomial, float]):
    def __contains__(self, item: CommutativePolynomialElement | float) -> bool: ...
    def get_canonical(
        self, monomial: CommutativePolynomialElement | float
    ) -> tuple[CommutativeMonomial, Canonicality, Realness]: ...

@final
class ComplexValuedCommutativeMomentMatrix(MomentMatrix[CommutativeMonomial, complex]):
    def __contains__(self, item: CommutativePolynomialElement | float) -> bool: ...
    def get_canonical(
        self, monomial: CommutativePolynomialElement | float
    ) -> tuple[CommutativeMonomial, Canonicality, Realness]: ...

@final
class RealValuedNonCommutativeMomentMatrix(MomentMatrix[NonCommutativeMonomial, float]):
    def __contains__(self, item: NonCommutativePolynomialElement | float) -> bool: ...
    def get_canonical(
        self, monomial: NonCommutativePolynomialElement | float
    ) -> tuple[NonCommutativeMonomial, Canonicality, Realness]: ...

@final
class ComplexValuedNonCommutativeMomentMatrix(MomentMatrix[NonCommutativeMonomial, complex]):
    def __contains__(self, item: NonCommutativePolynomialElement | float) -> bool: ...
    def get_canonical(
        self, monomial: NonCommutativePolynomialElement | float
    ) -> tuple[NonCommutativeMonomial, Canonicality, Realness]: ...

class BaseSdpRelaxation(Generic[MonomialType, Scalar]):
    @property
    def objective(self) -> Polynomial[MonomialType, Scalar]: ...
    # `dict` and `list` are invariant, so these three are declared as the read-only views they are
    # only ever used as, letting each concrete class below hand back its own moment matrix class.
    # Every other container below already names an exact type through `MonomialType`/`Scalar`.
    @property
    def moment_matrices(self) -> Mapping[int, MomentMatrix[MonomialType, Scalar]]: ...
    @property
    def localising_moment_matrices_inequalities(
        self,
    ) -> Mapping[int, list[MomentMatrix[MonomialType, Scalar]]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> Mapping[
        int, list[tuple[Polynomial[MonomialType, Scalar], list[Polynomial[MonomialType, Scalar]], Hermiticity]]
    ]: ...
    @property
    def moment_equalities(self) -> list[tuple[Polynomial[MonomialType, Scalar], Scalar]]: ...
    @property
    def moment_inequalities(self) -> list[tuple[Polynomial[MonomialType, Scalar], float]]: ...
    @property
    def is_real(self) -> bool: ...
    @overload
    def rewrite(self, mon_or_poly: MonomialType) -> MonomialType: ...
    @overload
    def rewrite(self, mon_or_poly: Polynomial[MonomialType, Scalar]) -> Polynomial[MonomialType, Scalar]: ...
    def get_coefficients_by_canonical(
        self, polynomial: Polynomial[MonomialType, Scalar]
    ) -> tuple[
        dict[MonomialType, Scalar],
        dict[MonomialType, tuple[complex, complex]],
    ]: ...
    def change_variables(
        self,
        polynomial: Polynomial[MonomialType, Scalar],
        mapping: dict[MonomialType, Any],
    ) -> Any: ...
    @property
    def generating_sets(self) -> dict[int, list[MonomialType]]: ...
    @property
    def equalities(
        self,
    ) -> dict[int, list[tuple[Polynomial[MonomialType, Scalar], list[MonomialType]]]]: ...
    @property
    def inequalities(
        self,
    ) -> dict[int, list[tuple[Polynomial[MonomialType, Scalar], list[MonomialType]]]]: ...

# Unlike every other class here, the four relaxations are `#[pyclass(subclass)]`, so Python code
# may subclass them and they are deliberately not `@final`.
class RealValuedCommutativeSdpRelaxation(BaseSdpRelaxation[CommutativeMonomial, float]):
    @property
    def is_real(self) -> Literal[True]: ...
    @property
    def objective(self) -> RealCoefficientsCommutativePolynomial: ...
    @property
    def moment_matrices(self) -> dict[int, RealValuedCommutativeMomentMatrix]: ...
    @property
    def localising_moment_matrices_inequalities(self) -> dict[int, list[RealValuedCommutativeMomentMatrix]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> Mapping[
        int,
        list[tuple[Polynomial[CommutativeMonomial, float], list[Polynomial[CommutativeMonomial, float]], Hermiticity]],
    ]: ...
    @overload
    def rewrite(self, mon_or_poly: CommutativePolynomialElement | float) -> CommutativeMonomial: ...
    @overload
    def rewrite(self, mon_or_poly: Polynomial[CommutativeMonomial, float]) -> RealCoefficientsCommutativePolynomial: ...

class ComplexValuedCommutativeSdpRelaxation(BaseSdpRelaxation[CommutativeMonomial, complex]):
    @property
    def is_real(self) -> Literal[False]: ...
    @property
    def objective(self) -> ComplexCoefficientsCommutativePolynomial: ...
    @property
    def moment_matrices(self) -> dict[int, ComplexValuedCommutativeMomentMatrix]: ...
    @property
    def localising_moment_matrices_inequalities(self) -> dict[int, list[ComplexValuedCommutativeMomentMatrix]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> Mapping[
        int,
        list[
            tuple[Polynomial[CommutativeMonomial, complex], list[Polynomial[CommutativeMonomial, complex]], Hermiticity]
        ],
    ]: ...
    @overload
    def rewrite(self, mon_or_poly: CommutativePolynomialElement | float) -> CommutativeMonomial: ...
    @overload
    def rewrite(
        self, mon_or_poly: Polynomial[CommutativeMonomial, complex]
    ) -> ComplexCoefficientsCommutativePolynomial: ...

class RealValuedNonCommutativeSdpRelaxation(BaseSdpRelaxation[NonCommutativeMonomial, float]):
    @property
    def is_real(self) -> Literal[True]: ...
    @property
    def objective(self) -> RealCoefficientsNonCommutativePolynomial: ...
    @property
    def moment_matrices(self) -> dict[int, RealValuedNonCommutativeMomentMatrix]: ...
    @property
    def localising_moment_matrices_inequalities(self) -> dict[int, list[RealValuedNonCommutativeMomentMatrix]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> Mapping[
        int,
        list[
            tuple[
                Polynomial[NonCommutativeMonomial, float], list[Polynomial[NonCommutativeMonomial, float]], Hermiticity
            ]
        ],
    ]: ...
    @overload
    def rewrite(self, mon_or_poly: NonCommutativePolynomialElement | float) -> NonCommutativeMonomial: ...
    @overload
    def rewrite(
        self, mon_or_poly: Polynomial[NonCommutativeMonomial, float]
    ) -> RealCoefficientsNonCommutativePolynomial: ...

class ComplexValuedNonCommutativeSdpRelaxation(BaseSdpRelaxation[NonCommutativeMonomial, complex]):
    @property
    def is_real(self) -> Literal[False]: ...
    @property
    def objective(self) -> ComplexCoefficientsNonCommutativePolynomial: ...
    @property
    def moment_matrices(self) -> dict[int, ComplexValuedNonCommutativeMomentMatrix]: ...
    @property
    def localising_moment_matrices_inequalities(self) -> dict[int, list[ComplexValuedNonCommutativeMomentMatrix]]: ...
    @property
    def localising_moment_matrices_equalities(
        self,
    ) -> Mapping[
        int,
        list[
            tuple[
                Polynomial[NonCommutativeMonomial, complex],
                list[Polynomial[NonCommutativeMonomial, complex]],
                Hermiticity,
            ]
        ],
    ]: ...
    @overload
    def rewrite(self, mon_or_poly: NonCommutativePolynomialElement | float) -> NonCommutativeMonomial: ...
    @overload
    def rewrite(
        self, mon_or_poly: Polynomial[NonCommutativeMonomial, complex]
    ) -> ComplexCoefficientsNonCommutativePolynomial: ...

@overload
def get_relaxation(
    variables: list[CommutativeOperator],
    level: int,
    objective: RealCoefficientsCommutativePolynomial | CommutativePolynomialElement,
    *,
    substitutions: dict[CommutativeMonomial, float | CommutativePolynomialElement] | None = None,
    operator_constraints: list[
        RealCoefficientsCommutativeConstraint
        | tuple[RealCoefficientsCommutativeConstraint, list[CommutativePolynomialElement | float] | None]
    ]
    | None = None,
    moment_constraints: list[RealCoefficientsCommutativeConstraint] | None = None,
    normalization_constraints: list[RealCoefficientsCommutativeConstraint] | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    extra_monomials: list[CommutativePolynomialElement | float] | None = None,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    check_uniqueness_with_length: bool = True,
) -> RealValuedCommutativeSdpRelaxation: ...
@overload
def get_relaxation(
    variables: list[CommutativeOperator],
    level: int,
    objective: RealCoefficientsCommutativePolynomial
    | ComplexCoefficientsCommutativePolynomial
    | CommutativePolynomialElement,
    *,
    substitutions: dict[CommutativeMonomial, float | CommutativePolynomialElement] | None = None,
    operator_constraints: list[
        RealCoefficientsCommutativeConstraint
        | ComplexCoefficientsCommutativeConstraint
        | tuple[
            RealCoefficientsCommutativeConstraint | ComplexCoefficientsCommutativeConstraint,
            list[CommutativePolynomialElement | float] | None,
        ]
    ]
    | None = None,
    moment_constraints: list[RealCoefficientsCommutativeConstraint | ComplexCoefficientsCommutativeConstraint]
    | None = None,
    normalization_constraints: list[RealCoefficientsCommutativeConstraint | ComplexCoefficientsCommutativeConstraint]
    | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    extra_monomials: list[CommutativePolynomialElement | float] | None = None,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    check_uniqueness_with_length: bool = True,
) -> ComplexValuedCommutativeSdpRelaxation: ...
@overload
def get_relaxation(
    variables: list[NonCommutativeOperator],
    level: int,
    objective: RealCoefficientsNonCommutativePolynomial | NonCommutativePolynomialElement,
    *,
    substitutions: dict[NonCommutativeMonomial, float | NonCommutativePolynomialElement] | None = None,
    operator_constraints: list[
        RealCoefficientsNonCommutativeConstraint
        | tuple[RealCoefficientsNonCommutativeConstraint, list[NonCommutativePolynomialElement | float] | None]
    ]
    | None = None,
    moment_constraints: list[RealCoefficientsNonCommutativeConstraint] | None = None,
    normalization_constraints: list[RealCoefficientsNonCommutativeConstraint] | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    extra_monomials: list[NonCommutativePolynomialElement | float] | None = None,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    check_uniqueness_with_length: bool = True,
) -> RealValuedNonCommutativeSdpRelaxation: ...
@overload
def get_relaxation(
    variables: list[NonCommutativeOperator],
    level: int,
    objective: RealCoefficientsNonCommutativePolynomial
    | ComplexCoefficientsNonCommutativePolynomial
    | NonCommutativePolynomialElement,
    *,
    substitutions: dict[NonCommutativeMonomial, float | NonCommutativePolynomialElement] | None = None,
    operator_constraints: list[
        RealCoefficientsNonCommutativeConstraint
        | ComplexCoefficientsNonCommutativeConstraint
        | tuple[
            RealCoefficientsNonCommutativeConstraint | ComplexCoefficientsNonCommutativeConstraint,
            list[NonCommutativePolynomialElement | float] | None,
        ]
    ]
    | None = None,
    moment_constraints: list[RealCoefficientsNonCommutativeConstraint | ComplexCoefficientsNonCommutativeConstraint]
    | None = None,
    normalization_constraints: list[
        RealCoefficientsNonCommutativeConstraint | ComplexCoefficientsNonCommutativeConstraint
    ]
    | None = None,
    substitution_strategy: RewritingStrategy = RewritingStrategy.Greedy,
    extra_monomials: list[NonCommutativePolynomialElement | float] | None = None,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    check_uniqueness_with_length: bool = True,
) -> ComplexValuedNonCommutativeSdpRelaxation: ...
