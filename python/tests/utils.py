from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ncpoleon._typing import MonomialType, RealOrComplexMatrix, Scalar
from ncpoleon.solve.solution import BaseSolution
from ncpoleon.solve.sos_decomposition import SoSDecomposition
from ncpoleon.utils import is_mosek_available

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation

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


def _reduce_sos_decomposition(
    sos: SoSDecomposition[MonomialType, Scalar],
) -> Polynomial[MonomialType, Scalar]:
    res = sum([poly.adjoint() * poly for poly in sos.moment_matrix_term.decomposition])

    for localizing_term in sos.hermitian_equalities_terms:
        res -= sum(
            [poly.adjoint() * localizing_term.generator * poly for poly in localizing_term.decomposition_negative]
        )
        res += sum(
            [poly.adjoint() * localizing_term.generator * poly for poly in localizing_term.decomposition_positive]
        )

    for localizing_term in sos.inequalities_terms:
        res += sum([poly.adjoint() * localizing_term.generator * poly for poly in localizing_term.decomposition])

    for moment_decomposition in (
        sos.moment_inequalities_terms + sos.moment_equalities_terms + sos.nonhermitian_equalities_terms
    ):
        res += moment_decomposition.term

    return res


def _build_moment_matrix_from_moments(
    generating_set: list[MonomialType],
    poly: Polynomial[MonomialType, Scalar],
    relaxation: BaseSdpRelaxation[MonomialType, Scalar],
    mapping: dict[MonomialType, Scalar],
) -> RealOrComplexMatrix:
    res = np.empty((len(generating_set), len(generating_set)), dtype=complex)

    for index_m, m in enumerate(generating_set):
        for index_n, n in enumerate(generating_set):
            res[index_m, index_n] = relaxation.rewrite(m.adjoint() * poly * n).change_variables(mapping)

    return res


def consistency_check(
    relaxation: BaseSdpRelaxation[MonomialType, Scalar],
    solution: BaseSolution,
    *,
    objective_sense: str,
    rtol: float = 1e-5,
    atol: float = 1e-07,
    sos_tol: float = 1e-04,
):
    sos_decomposition = 0
    mapping = {
        relaxation.rewrite(m.adjoint() * n): solution[relaxation.rewrite(m.adjoint() * n)]
        for mm_id, generating_set in relaxation.generating_sets.items()
        for m in generating_set
        for n in generating_set
    }

    for mm_id, generating_set in relaxation.generating_sets.items():
        mm_matrix_from_moments = _build_moment_matrix_from_moments(generating_set, 1, relaxation, mapping)
        # Check that the moments are indeed those in the moment matrix
        moment_matrix = solution.moment_matrix_by_mm_id[mm_id]
        np.testing.assert_allclose(moment_matrix, mm_matrix_from_moments, rtol=rtol, atol=atol)
        eigvals = np.linalg.eigvalsh(moment_matrix)
        # Check that the moment matrix is PSD
        np.testing.assert_allclose(eigvals[eigvals < 0], 0, rtol=rtol, atol=atol)

        for polynomial, matrix, generating_set in solution.localizing_matrices_inequality_by_mm_id.get(mm_id, []):
            # Check that the moments are indeed those in the localizing moment matrix
            np.testing.assert_allclose(
                matrix,
                _build_moment_matrix_from_moments(generating_set, polynomial, relaxation, mapping),
                rtol=rtol,
                atol=atol,
            )
            eigvals = np.linalg.eigvalsh(matrix)
            # Check that the localizing moment matrix is PSD
            np.testing.assert_allclose(eigvals[eigvals < 0], 0, rtol=rtol, atol=atol)

        for polynomial, generating_set in relaxation.equalities.get(mm_id, []):
            # Check that the moments in the localizing moment matrices are nil
            np.testing.assert_allclose(
                _build_moment_matrix_from_moments(generating_set, polynomial, relaxation, mapping),
                0,
                rtol=rtol,
                atol=atol,
            )

        sos_decomposition += _reduce_sos_decomposition(solution.get_sos_decomposition_by_mm_id()[mm_id])

    if objective_sense == "max":
        sos_decomposition *= -1

    # Check that the SoS decomposition reduces to the objective
    assert relaxation.rewrite(relaxation.objective - sos_decomposition).is_zero(sos_tol)

    # Check that the moment equalities are satisfied
    for polynomial, scalar in relaxation.moment_equalities:
        np.testing.assert_allclose(polynomial.change_variables(mapping), scalar, rtol=rtol, atol=atol)

    # Check that the moment inequalities are satisfied
    for polynomial, scalar in relaxation.moment_inequalities:
        actual = complex(polynomial.change_variables(mapping))
        np.testing.assert_allclose(actual.imag, 0, rtol=rtol, atol=atol)
        assert actual.real >= scalar or np.testing.assert_allclose(actual.real, scalar, rtol=rtol, atol=atol) is None
