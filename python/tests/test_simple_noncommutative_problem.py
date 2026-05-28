import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation
from ncpoleon.export import to_mosek, to_picos
from ncpoleon.utils import is_mosek_available

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_noncommutative_parameters():
    for export in ["picos", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 1 / 8)]:
            marks = []

            if export == "mosek":
                marks.append(
                    pytest.mark.skipif(
                        not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
                    )
                )

            yield pytest.param(export, level, expected, marks=marks)


def generate_simple_noncommutative_with_substitution_parameters():
    for export in ["picos", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 2.15e-05)]:
            marks = []

            if export == "mosek":
                marks.append(
                    pytest.mark.skipif(
                        not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
                    )
                )

            yield pytest.param(export, level, expected, marks=marks)


@pytest.mark.parametrize("export, level, expected", generate_simple_noncommutative_parameters())
def test_simple_real_noncommutative_problem(export: str, level: int, expected: float):
    x1, x2 = generate_noncommutative_variables("x", 2, starting_index=1, hermitian=True)
    obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]

    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)

    if export == "picos":
        problem = to_picos(sdp, "max", primal=True, verbosity=0)
        problem.solve()
        assert problem.value == pytest.approx(expected)
    elif export == "mosek":
        problem = to_mosek(sdp, "max", primal=True)
        problem.solve()
        assert problem.primalObjValue() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("export, level, expected", generate_simple_noncommutative_with_substitution_parameters())
def test_simple_real_noncommutative_problem_with_commutative_substitution(export: str, level: int, expected: float):
    x1, x2 = generate_noncommutative_variables("x", 2, starting_index=1, hermitian=True)
    obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    substitutions = {x2 * x1: x1 * x2}

    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints, substitutions=substitutions)

    if export == "picos":
        problem = to_picos(sdp, "max", primal=True, verbosity=0)
        problem.solve()
        assert problem.value == pytest.approx(expected, abs=1e-6)
    elif export == "mosek":
        problem = to_mosek(sdp, "max", primal=True)
        problem.solve()
        assert problem.primalObjValue() == pytest.approx(expected, abs=1e-6)
