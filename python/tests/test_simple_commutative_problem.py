from math import sqrt

import pytest
from ncpoleon import generate_commutative_variables, get_relaxation, solve
from ncpoleon.utils import is_mosek_available

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_commutative_parameters():
    for solver in ["picos", "mosek"]:
        for level, expected in [(1, -0.5), (2, 1 - sqrt(2))]:
            marks = []

            if solver == "mosek":
                marks.append(
                    pytest.mark.skipif(
                        not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
                    )
                )

            yield pytest.param(solver, level, expected, marks=marks)


@pytest.mark.parametrize("solver, level, expected", generate_simple_commutative_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_commutative_problem(solver: str, level: int, expected: float, force_primal: bool):
    x0 = generate_commutative_variables("x", 1, projector=True)[0]
    x1 = generate_commutative_variables("x", 1, real=True, starting_index=1)[0]
    obj = 2 * x0 * x1
    operator_constraints = [-(x1**2) + x1 + 1 / 4 >= 0]

    sdp = get_relaxation([x0, x1], level, obj, operator_constraints=operator_constraints)
    solution = solve(sdp, "min", force_primal=force_primal, solver=solver)
    assert solution.value == pytest.approx(expected)
    # TODO: test other attributes here
