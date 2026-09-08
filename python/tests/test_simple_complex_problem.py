from math import sqrt

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve

from .utils import SOLVER_SKIPS, SOLVERS, consistency_check

# The moment constraints exercised below, keyed by the name used in the parametrization ids. The polynomials need the
# variables, so `_moment_constraints` builds them per test rather than storing them here.
MOMENT_INEQUALITY_CASES = [("self_adjoint_sum", -0.5), ("imaginary_commutator", -2.0)]
MOMENT_EQUALITY_CASES = [
    ("self_adjoint_monomial", -sqrt(2)),
    ("hermitian_real_bound", -sqrt(15) / 2),
    ("antihermitian_imaginary_bound", -sqrt(15) / 2),
]

# Every case above, plus the unconstrained problem at both levels, crossed with both solvers and both forms.
SOS_DECOMPOSITION_CASES = [("no_moment_constraint", 1), ("no_moment_constraint", 2)] + [
    (case, 1) for case, _expected in MOMENT_INEQUALITY_CASES + MOMENT_EQUALITY_CASES
]


def generate_simple_complex_parameters():
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for level, expected in [(1, -2.0), (2, -2.0)]:
            res.append(pytest.param(solver, level, expected, marks=[SOLVER_SKIPS[solver]]))

    return res


def generate_moment_constraint_parameters(cases: list[tuple[str, float]]):
    """Cross a list of `(case, expected)` pairs with both solvers and both problem forms."""
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for case, expected in cases:
            for force_primal in [True, False]:
                res.append(pytest.param(solver, case, expected, force_primal, marks=[SOLVER_SKIPS[solver]]))

    return res


def generate_sos_decomposition_parameters():
    """Same cases as the value tests, but asserting the sum-of-squares certificate rather than the optimum."""
    res = []

    for solver in ["picos-cvxopt", "mosek"]:
        for case, level in SOS_DECOMPOSITION_CASES:
            for force_primal in [True, False]:
                res.append(pytest.param(solver, case, level, force_primal, marks=[SOLVER_SKIPS[solver]]))

    return res


def _simple_complex_params():
    """A complex-valued relaxation: the commutator constraint carries imaginary coefficients.

    `min <x1x2 + x2x1>` over `|x1|, |x2| <= 1` is `-2`, and the commutator constraint does not bind.
    """
    (x1, x2), identity = generate_noncommutative_variables(
        "x", 2, starting_index=1, hermitian=True, return_identity=True
    )
    obj = x1 * x2 + x2 * x1
    operator_constraints = [identity - x1**2 >= 0, identity - x2**2 >= 0, 1j * (x1 * x2 - x2 * x1) >= 0]
    return x1, x2, obj, operator_constraints


def _simple_real_params():
    x1, x2 = generate_noncommutative_variables("x", 2, starting_index=1, hermitian=True)
    obj = x1 * x2 + x2 * x1
    operator_constraints = [1 - x1**2 >= 0, 1 - x2**2 >= 0]
    return x1, x2, obj, operator_constraints


def _moment_constraints(case: str, x1, x2):
    """The moment constraint named by `case`, built against the variables of the problem under test."""
    return {
        "no_moment_constraint": [],
        # Hermitian polynomials over non-self-adjoint monomials
        "self_adjoint_sum": [x1 * x2 + x2 * x1 >= -0.5],
        "imaginary_commutator": [1j * (x1 * x2 - x2 * x1) >= -0.5],
        "self_adjoint_monomial": [x1**2 == 0.5],
        # The same constraint written two ways, one hermitian and one antihermitian
        "hermitian_real_bound": [1j * (x1 * x2 - x2 * x1) == 0.5],
        "antihermitian_imaginary_bound": [x1 * x2 - x2 * x1 == -0.5j],
    }[case]


@pytest.mark.parametrize("level", [1, 2])
def test_simple_complex_problem_relaxation(benchmark, level):
    x1, x2, obj, operator_constraints = _simple_complex_params()
    benchmark(get_relaxation, [x1, x2], level, obj, operator_constraints=operator_constraints)


@pytest.mark.parametrize("solver, level, expected", generate_simple_complex_parameters())
@pytest.mark.parametrize("force_primal", [True, False])
def test_simple_complex_problem(benchmark, solver: str, level: int, expected: float, force_primal: bool):
    x1, x2, obj, operator_constraints = _simple_complex_params()
    sdp = get_relaxation([x1, x2], level, obj, operator_constraints=operator_constraints)
    sol = benchmark(solve, sdp, "min", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="min", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, case, expected, force_primal", generate_moment_constraint_parameters(MOMENT_INEQUALITY_CASES)
)
def test_complex_problem_with_hermitian_moment_inequality(
    benchmark, solver: str, case: str, expected: float, force_primal: bool
):
    """A moment inequality whose polynomial is hermitian but built from non-self-adjoint monomials.

    The dual exports used to assert that no such monomial appeared in a moment inequality, which is false for any
    hermitian polynomial that is not a combination of self-adjoint monomials.
    """
    x1, x2, obj, operator_constraints = _simple_complex_params()
    sdp = get_relaxation(
        [x1, x2],
        1,
        obj,
        operator_constraints=operator_constraints,
        moment_constraints=_moment_constraints(case, x1, x2),
    )
    sol = benchmark(solve, sdp, "min", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="min", sos_tol=1e-07)


@pytest.mark.parametrize(
    "solver, case, expected, force_primal", generate_moment_constraint_parameters(MOMENT_EQUALITY_CASES)
)
def test_complex_problem_with_moment_equality(benchmark, solver: str, case: str, expected: float, force_primal: bool):
    """`hermitian_real_bound` and `antihermitian_imaginary_bound` are the same constraint written two ways.

    `<i(x1x2 - x2x1)> == 1/2` and `<x1x2 - x2x1> == -i/2` differ only by the factor `i`, so they must agree. The
    antihermitian form drives the dual multiplier off the real axis.
    """
    x1, x2, obj, operator_constraints = _simple_complex_params()
    sdp = get_relaxation(
        [x1, x2],
        1,
        obj,
        operator_constraints=operator_constraints,
        moment_constraints=_moment_constraints(case, x1, x2),
    )
    sol = benchmark(solve, sdp, "min", force_primal=force_primal, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(sdp, sol, objective_sense="min", sos_tol=1e-07)


def test_non_hermitian_moment_inequality_is_rejected():
    """A moment inequality has a real bound, so ordering a non-hermitian moment against it is meaningless."""
    x1, x2, obj, operator_constraints = _simple_complex_params()

    with pytest.raises(ValueError, match="isn't Hermitian"):
        get_relaxation(
            [x1, x2], 1, obj, operator_constraints=operator_constraints, moment_constraints=[x1 * x2 >= -0.5]
        )


@pytest.mark.parametrize(
    "case, message",
    [
        ("hermitian_monomial", "hermitian and thus real-valued"),
        ("hermitian_polynomial", "hermitian and thus real-valued"),
        ("antihermitian_monomial", "antihermitian and thus purely imaginary"),
        ("antihermitian_polynomial", "antihermitian and thus purely imaginary"),
    ],
)
def test_unsatisfiable_moment_equality_is_rejected(case: str, message: str):
    """A hermitian moment is real and an antihermitian one is purely imaginary, whatever the state.

    A bound that contradicts either is unsatisfiable by construction, and saying so beats letting the solver report an
    unusable status.
    """
    x1, x2, obj, operator_constraints = _simple_complex_params()
    moment_constraints = {
        "hermitian_monomial": [x1**2 == 0.5 + 1j],
        "hermitian_polynomial": [1j * (x1 * x2 - x2 * x1) == 0.5 + 1j],
        "antihermitian_monomial": [1j * x1**2 == 0.5 + 1j],
        "antihermitian_polynomial": [x1 * x2 - x2 * x1 == 0.5],
    }[case]

    with pytest.raises(ValueError, match=message):
        get_relaxation(
            [x1, x2], 1, obj, operator_constraints=operator_constraints, moment_constraints=moment_constraints
        )


def test_unsatisfiable_moment_equality_is_rejected_on_a_real_problem():
    """The moment matrix of a real-valued problem is symmetric, so an antihermitian moment is identically zero."""
    x1, x2, obj, operator_constraints = _simple_real_params()

    with pytest.raises(ValueError, match="antihermitian and thus purely imaginary"):
        get_relaxation(
            [x1, x2],
            1,
            obj,
            operator_constraints=operator_constraints,
            moment_constraints=[x1 * x2 - x2 * x1 == 0.5],
        )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("force_primal", [True, False])
def test_satisfiable_counterparts_still_build(solver: str, force_primal: bool):
    """The mirror of the rejection cases: the same shapes with a compatible bound must still solve."""
    x1, x2, obj, operator_constraints = _simple_complex_params()

    for moment_constraints, expected in [
        ([x1**2 == 0.5], -sqrt(2)),
        ([1j * x1**2 == 1j], -2.0),
        ([x1 * x2 - x2 * x1 == 0j], -2.0),
    ]:
        sdp = get_relaxation(
            [x1, x2], 1, obj, operator_constraints=operator_constraints, moment_constraints=moment_constraints
        )
        sol = solve(sdp, "min", force_primal=force_primal, solver=solver)
        assert sol.value == pytest.approx(expected, abs=1e-6)
        consistency_check(sdp, sol, objective_sense="min", sos_tol=1e-07)
