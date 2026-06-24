from math import sqrt

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve
from ncpoleon.utils import is_mosek_available

from .utils import reduce_sos_decomposition


@pytest.mark.parametrize(
    "solver",
    [
        "picos-cvxopt",
        pytest.param(
            "mosek",
            marks=pytest.mark.skipif(
                not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
            ),
        ),
    ],
)
@pytest.mark.parametrize("use_primal", [False, True])
# TODO: we should use the benchmark fixture along with a parameter indicating which part (relaxation/solving/sos) we're
#  benchmarking
@pytest.mark.benchmark
def test_chsh_uniform(solver, use_primal):
    """
    What is the largest CHSH value possible if we know that the inputs (x,y) = (0,0)
    produce a uniform distribution?

    Representing the problem via observables we have

    max Tr[rho (A0 otimes (B0 + B1) + A1 otimes (B0 - B1))]
    s.t. Ax^2 = I, By^2 = I,
        Tr[rho (A0 otimes B0)] = 0
        Tr[rho (A0 otimes I)] = 0
        Tr[rho (I otimes B0)] = 0

    Correct answer is 3sqrt(3)/2. Solves at level 1.
    """
    level = 1

    A = generate_noncommutative_variables("A", 2, hermitian=True)
    B = generate_noncommutative_variables("B", 2, hermitian=True)

    substitutions = {b * a: a * b for a in A for b in B} | {x**2: 1 for x in A + B}
    obj = A[0] * (B[0] + B[1]) + A[1] * (B[0] - B[1])
    moment_constraints = [A[0] * B[0] == 0, A[0] == 0, B[0] == 0]

    sdp = get_relaxation(A + B, level, obj, substitutions=substitutions, moment_constraints=moment_constraints)
    sol = solve(sdp, "max", force_primal=use_primal, solver=solver)

    assert sol.value == pytest.approx(3 * sqrt(3) / 2)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) + obj)).is_zero(1e-7)
