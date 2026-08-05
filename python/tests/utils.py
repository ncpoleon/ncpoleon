from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest
from ncpoleon._typing import PolynomialElements, Scalar
from ncpoleon.solve.sos_decomposition import SoSDecomposition
from ncpoleon.utils import is_mosek_available

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial

# Resolved once for the whole session. `is_mosek_available()` is deliberately uncached so
# that a license added mid-process is picked up, which means every caller pays for its own
# check and the skipif marks below are evaluated once per parametrized case while pytest
# expands the generators. On GitHub's macOS runners each call blocks for 70s, which is why
# we want to call it only once.
MOSEK_AVAILABLE = is_mosek_available()
MOSEK_SKIP_REASON = "Mosek is not installed or a Mosek license is not available."

# picos is a hard dependency of the `picos` extra, but cvxopt -- which picos pulls in --
# publishes no Windows arm64 wheel and cannot be built from its sdist against MSVC, so
# that platform installs neither. Gating on picos alone is enough: it is pure Python and
# cvxopt is one of its install requirements, so one present implies the other.
PICOS_AVAILABLE = find_spec("picos") is not None
PICOS_SKIP_REASON = "picos is not installed."

# One skip mark per solver, so a parameter generator can tag a case with `SOLVER_SKIPS[solver]`
# instead of repeating the availability check per branch.
SOLVER_SKIPS = {
    "picos-cvxopt": pytest.mark.skipif(not PICOS_AVAILABLE, reason=PICOS_SKIP_REASON),
    "mosek": pytest.mark.skipif(not MOSEK_AVAILABLE, reason=MOSEK_SKIP_REASON),
}

SOLVERS = [pytest.param(solver, marks=[mark]) for solver, mark in SOLVER_SKIPS.items()]


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
