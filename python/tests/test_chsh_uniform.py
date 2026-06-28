from math import sqrt

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve
from .utils import SOLVERS, reduce_sos_decomposition


def _chsh_variables():
    A = generate_noncommutative_variables("A", 2, hermitian=True)
    B = generate_noncommutative_variables("B", 2, hermitian=True)
    substitutions = {b * a: a * b for a in A for b in B} | {x**2: 1 for x in A + B}
    obj = A[0] * (B[0] + B[1]) + A[1] * (B[0] - B[1])
    moment_constraints = [A[0] * B[0] == 0, A[0] == 0, B[0] == 0]
    return A, B, substitutions, obj, moment_constraints


@pytest.fixture
def chsh_sdp():
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
    A, B, substitutions, obj, moment_constraints = _chsh_variables()
    sdp = get_relaxation(A + B, 1, obj, substitutions=substitutions, moment_constraints=moment_constraints)
    return sdp, obj


def test_chsh_uniform_relaxation(benchmark):
    A, B, substitutions, obj, moment_constraints = _chsh_variables()
    benchmark(get_relaxation, A + B, 1, obj, substitutions=substitutions, moment_constraints=moment_constraints)

@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("use_primal", [False, True])
def test_chsh_uniform_solve(benchmark, chsh_sdp, solver, use_primal):
    sdp, obj = chsh_sdp
    sol = benchmark(solve, sdp, "max", force_primal=use_primal, solver=solver)
    assert sol.value == pytest.approx(3 * sqrt(3) / 2)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) + obj)).is_zero(1e-7)
