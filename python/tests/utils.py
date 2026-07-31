from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ncpoleon._typing import PolynomialElements, Scalar
from ncpoleon.solve.sos_decomposition import SoSDecomposition
from ncpoleon.utils import is_mosek_available

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial

SOLVERS = [
    "picos-cvxopt",
    pytest.param(
        "mosek",
        marks=pytest.mark.skipif(
            not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
        ),
    ),
]


def reduce_sos_decomposition(
    sos: SoSDecomposition[PolynomialElements, Scalar],
) -> Polynomial[PolynomialElements, Scalar]:
    res = sum([poly.adjoint() * poly for poly in sos.moment_matrix_term.decomposition])

    for localizing_term in sos.equalities_terms:
        res -= sum(
            [poly.adjoint() * localizing_term.generator * poly for poly in localizing_term.decomposition_negative]
        )
        res += sum(
            [poly.adjoint() * localizing_term.generator * poly for poly in localizing_term.decomposition_positive]
        )

    for localizing_term in sos.inequalities_terms:
        res += sum([poly.adjoint() * localizing_term.generator * poly for poly in localizing_term.decomposition])

    for moment_decomposition in sos.moment_equalities_terms + sos.moment_inequalities_terms:
        res += moment_decomposition.coefficient * moment_decomposition.generator

    return res
