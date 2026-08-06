from math import sqrt

import pytest
from ncpoleon import generate_commutative_variables, get_relaxation, solve

from .utils import SOLVER_SKIPS, reduce_sos_decomposition

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_commutative_parameters():
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, -0.5), (2, 1 - sqrt(2))]:
            res.append(pytest.param(solver, level, expected, marks=[SOLVER_SKIPS[solver]]))

    return res


def _simple_commutative_params():
    x0 = generate_commutative_variables("x", 1, projector=True)[0]
    x1 = generate_commutative_variables("x", 1, real=True, starting_index=1)[0]
    obj = 2 * x0 * x1
    operator_constraints = [-(x1**2) + x1 + 1 / 4 >= 0]
    return x0, x1, obj, operator_constraints


@pytest.mark.parametrize("level", [1, 2])
def test_simple_real_commutative_problem_relaxation(benchmark, level):
    x0, x1, obj, operator_constraints = _simple_commutative_params()
    benchmark(get_relaxation, [x0, x1], level, obj, operator_constraints=operator_constraints)


@pytest.mark.parametrize("solver, level, expected", generate_simple_commutative_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_commutative_problem(benchmark, solver: str, level: int, expected: float, force_primal: bool):
    x0, x1, obj, operator_constraints = _simple_commutative_params()
    sdp = get_relaxation([x0, x1], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "min", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) - obj)).is_zero(1e-7)


@pytest.mark.parametrize("solver, level, expected", generate_simple_commutative_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_noncommutative_problem_with_extra_monomials(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    (x0, x1), identity = generate_commutative_variables("x", 2, real=True, return_identity=True)
    obj = 2 * x0 * x1

    if level == 1:
        extra_monomials = [identity, x0, x1]
        operator_constraints = [(-(x1**2) + x1 + 1 / 4 >= 0, [identity])]
    elif level == 2:
        extra_monomials = [identity, x0, x1, x0 * x1, x1**2]
        operator_constraints = [(-(x1**2) + x1 + 1 / 4 >= 0, [identity, x0, x1])]

    sdp = benchmark(
        get_relaxation,
        [x0, x1],
        level=-1,
        objective=obj,
        substitutions={x0**2: x0},
        operator_constraints=operator_constraints,
        extra_monomials=extra_monomials,
        normalization_constraints=[identity == 1],
    )
    sol = solve(sdp, "min", verbosity=0, solver=solver, force_primal=force_primal)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) - obj)).is_zero(1e-7)
