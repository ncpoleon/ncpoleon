from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial

from ncpoleon._typing import MonomialType, Scalar


@dataclass(eq=False, order=False, kw_only=True)
class MomentMatrixDecomposition(Generic[MonomialType, Scalar]):
    """A single moment matrix contribution to a SoS decomposition.

    Docs TODO:
    """

    decomposition: list[Polynomial[MonomialType, Scalar]]


@dataclass(eq=False, order=False, kw_only=True)
class LocalizingMomentMatrixInequalityDecomposition(Generic[MonomialType, Scalar]):
    """A single localizing moment matrix inequality contribution to a SoS decomposition.

    Docs TODO:
    """

    generator: Polynomial[MonomialType, Scalar]
    decomposition: list[Polynomial[MonomialType, Scalar]]


@dataclass(eq=False, order=False, kw_only=True)
class LocalizingMomentMatrixEqualityDecomposition(Generic[MonomialType, Scalar]):
    """A single localizing moment matrix equality contribution to a SoS decomposition.

    Docs TODO:
    """

    term: Polynomial[MonomialType, Scalar]


@dataclass(eq=False, order=False, kw_only=True)
class SingleMomentEqualityDecomposition(Generic[MonomialType, Scalar]):
    """A single scalar moment contribution to a SoS decomposition.

    Docs TODO:
    """

    term: Polynomial[MonomialType, Scalar]


@dataclass(eq=False, order=False, kw_only=True)
class SingleMomentInequalityDecomposition(Generic[MonomialType, Scalar]):
    """A single scalar moment contribution to a SoS decomposition.

    Docs TODO:
    """

    term: Polynomial[MonomialType, Scalar]


@dataclass(eq=False, order=False, kw_only=True)
class SoSDecomposition(Generic[MonomialType, Scalar]):
    moment_matrix_term: MomentMatrixDecomposition[MonomialType, Scalar]
    equalities_terms: list[LocalizingMomentMatrixEqualityDecomposition[MonomialType, Scalar]]
    inequalities_terms: list[LocalizingMomentMatrixInequalityDecomposition[MonomialType, Scalar]]
    moment_equalities_terms: list[SingleMomentEqualityDecomposition[MonomialType, Scalar]]
    moment_inequalities_terms: list[SingleMomentInequalityDecomposition[MonomialType, Scalar]]
