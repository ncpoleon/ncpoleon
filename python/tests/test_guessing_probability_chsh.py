from math import sqrt

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve

from .utils import SOLVERS, reduce_sos_decomposition


def _guessing_probability_chsh_params(w):
    [M0, M1] = generate_noncommutative_variables("M", 2, projector=True)
    [N0, N1] = generate_noncommutative_variables("N", 2, projector=True)
    [E] = generate_noncommutative_variables("E", 1, projector=True)

    substitutions = {}
    for op1 in [M0, M1]:
        for op2 in [N0, N1]:
            substitutions[op2 * op1] = op1 * op2
        substitutions[E * op1] = op1 * E
    for op2 in [N0, N1]:
        substitutions[E * op2] = op2 * E

    A0 = 2 * M0 - 1
    A1 = 2 * M1 - 1
    B0 = 2 * N0 - 1
    B1 = 2 * N1 - 1
    moment_constraints = [A0 * (B0 + B1) + A1 * (B0 - B1) == w]
    obj = M0 * E + (1 - M0) * (1 - E)

    return [M0, M1, N0, N1, E], obj, substitutions, moment_constraints


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("w", [2.0, 2.25, 2.5])
def test_guessing_probability_chsh_relaxation(benchmark, level, w):
    """
    NCPOP relaxation of the guessing probability problem for DI Cryptography

    We want to maximize the probability that an adversary Eve can guess the outcome
    of Alice's measurement if we are promised that Alice and Bob's devices violate some
    Bell-inequality (here CHSH).

    max Tr[rho (A_{0|0} otimes E)]/2 + Tr[(A_{1|0} otimes (id - E))]
    s.t. CHSH = w

    Level 1 should give trivial value of 1 and level 2 should give the optimal value of (1 + sqrt(2 - (w**2)/4))/2
    """
    variables, obj, substitutions, moment_constraints = _guessing_probability_chsh_params(w)
    benchmark(get_relaxation, variables, level, obj, substitutions=substitutions, moment_constraints=moment_constraints)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("use_primal", [False, True])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("w", [2.0, 2.25, 2.5])
def test_guessing_probability_chsh_solve(benchmark, solver, use_primal, level, w):
    variables, obj, substitutions, moment_constraints = _guessing_probability_chsh_params(w)
    sdp = get_relaxation(variables, level, obj, substitutions=substitutions, moment_constraints=moment_constraints)
    sol = benchmark(solve, sdp, "max", solver=solver, force_primal=use_primal)

    if level == 1:
        assert sol.value == pytest.approx(1.0)
    else:
        assert sol.value == pytest.approx((1 + sqrt(2 - (w**2) / 4)) / 2)

    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) + obj)).is_zero(1e-7)
