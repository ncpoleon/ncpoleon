from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial

from ncpoleon._typing import PolynomialElements, Scalar


@dataclass(eq=False, order=False, kw_only=True)
class MomentMatrixDecomposition(Generic[PolynomialElements, Scalar]):
    """A single moment matrix contribution to a SoS decomposition.

    Docs TODO
    """

    decomposition: list[Polynomial[PolynomialElements, Scalar]]


@dataclass(eq=False, order=False, kw_only=True)
class LocalizingMomentMatrixInequalityDecomposition(Generic[PolynomialElements, Scalar]):
    """A single localizing moment matrix inequality contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    decomposition: list[Polynomial[PolynomialElements, Scalar]]


@dataclass(eq=False, order=False, kw_only=True)
class LocalizingMomentMatrixEqualityDecomposition(Generic[PolynomialElements, Scalar]):
    """A single localizing moment matrix equality contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    decomposition_positive: list[Polynomial[PolynomialElements, Scalar]]
    decomposition_negative: list[Polynomial[PolynomialElements, Scalar]]


@dataclass(eq=False, order=False, kw_only=True)
class SingleMomentEqualityDecomposition(Generic[PolynomialElements, Scalar]):
    """A single scalar moment contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    coefficient: Scalar

@dataclass(eq=False, order=False, kw_only=True)
class SingleMomentInequalityDecomposition(Generic[PolynomialElements, Scalar]):
    """A single scalar moment contribution to a SoS decomposition.

    Docs TODO
    """

    generator: Polynomial[PolynomialElements, Scalar]
    coefficient: float


@dataclass(eq=False, order=False, kw_only=True)
class SoSDecomposition(Generic[PolynomialElements, Scalar]):
    moment_matrix_term: MomentMatrixDecomposition[PolynomialElements, Scalar]
    equalities_terms: list[LocalizingMomentMatrixEqualityDecomposition[PolynomialElements, Scalar]]
    inequalities_terms: list[LocalizingMomentMatrixInequalityDecomposition[PolynomialElements, Scalar]]
    moment_equalities_terms: list[SingleMomentEqualityDecomposition[PolynomialElements, Scalar]]
    moment_inequalities_terms: list[SingleMomentInequalityDecomposition[PolynomialElements, Scalar]]
