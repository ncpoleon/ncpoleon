import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve

from .utils import SOLVER_SKIPS, consistency_check


def generate_simple_noncommutative_parameters():
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 1 / 8)]:
            res.append(pytest.param(solver, level, expected, marks=[SOLVER_SKIPS[solver]]))

    return res


def generate_simple_noncommutative_with_equality_constraints_parameters():
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 1 / 8)]:
            for force_primal in [True, False]:
                marks = [SOLVER_SKIPS[solver]]
                res.append(pytest.param(solver, level, expected, force_primal, marks=marks))

    return res


def generate_simple_noncommutative_with_substitution_parameters():
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 2.15e-05)]:
            res.append(pytest.param(solver, level, expected, marks=[SOLVER_SKIPS[solver]]))

    return res


def generate_simple_noncommutative_as_operator_equalities_parameters():
    """Parameters for imposing commutativity as an operator equality rather than as a substitution.

    A substitution rewrites every monomial, whereas an operator equality only constrains the moments its
    localising set reaches. The relaxation is therefore weaker, and its level 2 maximum correspondingly
    larger than the 2.15e-05 of the substitution formulation.
    """
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, 1 / 8), (2, 5.52e-05)]:
            for force_primal in [True, False]:
                res.append(pytest.param(solver, level, expected, force_primal, marks=[SOLVER_SKIPS[solver]]))

    return res


def _simple_noncommutative_vars(with_identity: bool = False):
    if with_identity:
        (x1, x2), identity = generate_noncommutative_variables(
            "x", 2, starting_index=1, hermitian=True, return_identity=True
        )
        obj = x2**2 - x1 * x2 / 2 - x2 * x1 / 2 - x2
        return x1, x2, obj, identity
    else:
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
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize("level", [1, 2])
def test_simple_real_noncommutative_problem_with_equality_constraints_relaxation(benchmark, level):
    x1, x2, obj = _simple_noncommutative_vars()
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
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, level, expected, force_primal", generate_simple_noncommutative_with_equality_constraints_parameters()
)
def test_simple_complex_noncommutative_problem_with_antihermitian_equality_constraints(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    # The factor `1j` makes the equality anti-hermitian and the relaxation complex, without changing its feasible set
    operator_constraints = [x2 - x2**2 == 0, 1j * (x1 - x1**2) == 0]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, level, expected, force_primal",
    [
        pytest.param(solver, level, expected, force_primal, marks=[SOLVER_SKIPS[solver]])
        for solver in ["picos-cvxopt", "mosek"]
        for level, expected in [(1, 1 / 8), (2, 0.0)]
        for force_primal in [True, False]
    ],
)
def test_simple_real_noncommutative_problem_with_nonhermitian_equality_constraints(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    # In a real relaxation, entries (i, j) and (j, i) of the localising matrix of this equality are the sum and the
    # difference of those of `x1 - x1**2` and of the commutator, so it imposes both `x1 - x1**2 == 0` and
    # `x2 * x1 == x1 * x2`
    operator_constraints = [x2 - x2**2 == 0, x1 - x1**2 + x2 * x1 - x1 * x2 == 0]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, level, expected, force_primal", generate_simple_noncommutative_with_equality_constraints_parameters()
)
def test_simple_complex_noncommutative_problem_with_nonhermitian_equality_constraints(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    # In a complex relaxation, the localising matrix of `h1 + 1j * h2` with hermitian `h1` and `h2` vanishes only if
    # both of theirs do, so this single equality imposes both `x1 - x1**2 == 0` and `x2 - x2**2 == 0`
    operator_constraints = [x1 - x1**2 + 1j * (x2 - x2**2) == 0]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


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
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0]
    substitutions = {x2 * x1: x1 * x2}
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints, substitutions=substitutions)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, level, expected, force_primal",
    generate_simple_noncommutative_as_operator_equalities_parameters(),
)
def test_simple_real_noncommutative_problem_with_commutative_substitution_as_antihermitian_operator_equalities(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    # The commutator is anti-hermitian, with real coefficients
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0, x2 * x1 == x1 * x2]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, level, expected, force_primal",
    generate_simple_noncommutative_as_operator_equalities_parameters(),
)
def test_simple_real_noncommutative_problem_with_commutative_substitution_as_hermitian_operator_equalities(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj = _simple_noncommutative_vars()
    operator_constraints = [x1 - x1**2 >= 0, x2 - x2**2 >= 0, 1j * (x2 * x1 - x1 * x2) == 0]
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "max", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)


@pytest.mark.parametrize("solver, level, expected", generate_simple_noncommutative_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_real_noncommutative_problem_with_extra_monomials(
    benchmark, solver: str, level: int, expected: float, force_primal: bool
):
    x1, x2, obj, identity = _simple_noncommutative_vars(with_identity=True)

    if level == 1:
        extra_monomials = [identity, x1, x2]
        operator_constraints = [(x1 - x1**2 >= 0, [identity]), (x2 - x2**2 >= 0, [identity])]
    elif level == 2:
        extra_monomials = [identity, x1, x2, x1**2, x1 * x2, x2 * x1, x2**2]
        operator_constraints = [(x1 - x1**2 >= 0, [identity, x1, x2]), (x2 - x2**2 >= 0, [identity, x1, x2])]

    sdp = benchmark(
        get_relaxation,
        [],
        level=-1,
        objective=obj,
        operator_constraints=operator_constraints,
        extra_monomials=extra_monomials,
        normalization_constraints=[identity == 1],
    )

    sol = solve(sdp, "max", verbosity=0, solver=solver, force_primal=force_primal)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="max", sos_tol=1e-07)
