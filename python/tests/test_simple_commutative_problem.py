from math import sqrt

import pytest
from ncpoleon import generate_commutative_variables, get_relaxation, solve
from ncpoleon.utils import is_mosek_available

from .utils import reduce_sos_decomposition

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_commutative_parameters():
    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, -0.5), (2, 1 - sqrt(2))]:
            if solver == "mosek":
                yield pytest.param(
                    solver,
                    level,
                    expected,
                    marks=[
                        pytest.mark.skipif(
                            not is_mosek_available(),
                            reason="Mosek is not installed or a Mosek license is not available.",
                        )
                    ],
                )

            elif solver == "picos-cvxopt":
                yield pytest.param(solver, level, expected)


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
