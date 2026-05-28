from math import sqrt

import pytest
from ncpoleon import generate_commutative_variables, get_relaxation
from ncpoleon.export import to_mosek, to_picos
from ncpoleon.utils import is_mosek_available

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_commutative_parameters():
    for export in ["picos", "mosek"]:
        for level, expected in [(1, -0.5), (2, 1 - sqrt(2))]:
            marks = []

            if export == "mosek":
                marks.append(
                    pytest.mark.skipif(
                        not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
                    )
                )

            yield pytest.param(export, level, expected, marks=marks)


@pytest.mark.parametrize("export, level, expected", generate_simple_commutative_parameters())
def test_simple_real_commutative_problem(export: str, level: int, expected: float):
    x0 = generate_commutative_variables("x", 1, projector=True)[0]
    x1 = generate_commutative_variables("x", 1, real=True, starting_index=1)[0]
    obj = 2 * x0 * x1
    operator_constraints = [-(x1**2) + x1 + 1 / 4 >= 0]

    sdp = get_relaxation([x0, x1], level, obj, operator_constraints=operator_constraints)

    if export == "picos":
        problem = to_picos(sdp, "min", primal=True, verbosity=0)
        problem.solve()
        assert problem.value == pytest.approx(expected)
    elif export == "mosek":
        problem = to_mosek(sdp, "min", primal=True)
        problem.solve()
        assert problem.primalObjValue() == pytest.approx(expected)
