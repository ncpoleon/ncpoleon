import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve
from ncpoleon.utils import is_mosek_available

from .utils import reduce_sos_decomposition

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_simple_noncommutative_parameters():
    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 1 / 8)]:
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


def generate_simple_noncommutative_with_equality_constraints_parameters():
    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 1 / 8)]:
            for force_primal in [True, False]:
                if solver == "mosek":
                    yield pytest.param(
                        solver,
                        level,
                        expected,
                        force_primal,
                        marks=[
                            pytest.mark.skipif(
                                not is_mosek_available(),
                                reason="Mosek is not installed or a Mosek license is not available.",
                            )
                        ],
                    )
                elif solver == "picos-cvxopt":
                    if level == 2 and force_primal:
                        yield pytest.param(
                            solver,
                            level,
                            expected,
                            force_primal,
                            marks=[
                                pytest.mark.xfail(
                                    reason="Solving the primal at level 2 using the CVXOPT Solver results in an error",
                                    raises=ArithmeticError,
                                )
                            ],
                        )
                    else:
                        yield pytest.param(solver, level, expected, force_primal)


def generate_simple_noncommutative_with_substitution_parameters():
    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 2.15e-05)]:
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


def _simple_noncommutative_vars():
    x1, x2 = generate_noncommutative_variables("x", 2, starting_index=1, hermitian=True)
    obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2
    return x1, x2, obj


@pytest.mark.parametrize("level", [1, 2])
def test_simple_real_noncommutative_problem_relaxation(benchmark, level):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    benchmark(get_relaxation, [x1, x2], level, obj, operator_constraints=operator_constraints)


@pytest.mark.parametrize("solver, level, expected", generate_simple_noncommutative_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_noncommutative_problem(benchmark, solver: str, level: int, expected: float, force_primal: bool):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) + obj)).is_zero(1e-7)


@pytest.mark.parametrize("level", [1, 2])
def test_simple_real_noncommutative_problem_with_equality_constraints_relaxation(benchmark, level):
    x1, x2, obj = _simple_noncommutative_vars()
    # FIXME: So, for SOME REASON, CVXOPT fails to solve the problem if we input the constraints in this order. That is,
    #  if we swap these two constraints, the code works. Maybe we'll have to investigate this at some point, but since
    #  it only happens on the primal, it's not *too* bad. It might reveal a bug on Picos' side though, so it might be
    #  worth invectigating
    operator_constraints = [x2 - x2**2 == 0, x1 - x1**2 == 0]
    benchmark(get_relaxation, [x1, x2], level, obj, operator_constraints=operator_constraints)


@pytest.mark.parametrize(
    "solver, level, expected, force_primal", generate_simple_noncommutative_with_equality_constraints_parameters()
)
def test_simple_real_noncommutative_problem_with_equality_constraints(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x2 - x2**2 == 0, x1 - x1**2 == 0]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) + obj)).is_zero(1e-7)


@pytest.mark.parametrize("level", [1, 2])
def test_simple_real_noncommutative_problem_with_commutative_substitution_relaxation(benchmark, level):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    substitutions = {x2 * x1: x1 * x2}
    benchmark(
        get_relaxation, [x1, x2], level, obj, operator_constraints=operator_constraints, substitutions=substitutions
    )


@pytest.mark.parametrize("solver, level, expected", generate_simple_noncommutative_with_substitution_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_noncommutative_problem_with_commutative_substitution(
    solver: str, level: int, expected: float, force_primal: bool, benchmark
):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    substitutions = {x2 * x1: x1 * x2}
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints, substitutions=substitutions)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    assert (sdp.rewrite(reduce_sos_decomposition(sol.get_sos_decomposition()) + obj)).is_zero(1e-7)
