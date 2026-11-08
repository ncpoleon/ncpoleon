"""Adding, removing or replacing constraints gives the relaxation built with the edited constraint list.

The one exception is that added moment constraints go after the normalization ones, which the build puts last.
"""

import pytest
from ncpoleon import generate_commutative_variables, generate_noncommutative_variables, get_relaxation

from .test_save_load import _snapshot


def _real_commutative():
    x0, x1 = generate_commutative_variables("x", 2, real=True)
    constraints = [x0 - x0**2 >= 0, x0 * x1 == 0, (x1 - x1**2 >= 0, [1, x1]), x0 + x1 - 1 == 0]
    replacements = [1 - x0**2 >= 0, x0 - x1 == 0]
    return (
        lambda constraints: get_relaxation([x0, x1], 2, 2 * x0 * x1, operator_constraints=constraints),
        constraints,
        replacements,
        x0,
    )


def _complex_noncommutative():
    a0, a1 = generate_noncommutative_variables("A", 2, hermitian=True)
    constraints = [a0 - a0**2 >= 0, 1j * (a1 * a0 - a0 * a1) == 0, (a1 - a1**2 >= 0, [1, a1]), a0 * a1 - a1 * a0 == 0]
    replacements = [1 - a0**2 >= 0, a0 - a1 == 0]
    return (
        lambda constraints: get_relaxation(
            [a0, a1], 2, (1 + 0j) * (a0 * a1 + a1 * a0), operator_constraints=constraints
        ),
        constraints,
        replacements,
        a0,
    )


PROBLEMS = pytest.mark.parametrize("problem", [_real_commutative, _complex_noncommutative])


@PROBLEMS
def test_add(problem):
    build, constraints, _, _ = problem()
    sdp = build(constraints[:1])
    sdp.add_operator_constraints(constraints[1:])
    assert _snapshot(sdp) == _snapshot(build(constraints))


@PROBLEMS
def test_remove(problem):
    build, (i0, e0, i1, e1), _, _ = problem()
    sdp = build([i0, e0, i1, e1])
    # The second equality and the first inequality
    sdp.remove_operator_constraints(equalities=[1], inequalities=[0])
    assert _snapshot(sdp) == _snapshot(build([e0, i1]))


@PROBLEMS
def test_replace_keeps_positions(problem):
    build, (i0, e0, i1, e1), (new_inequality, new_equality), _ = problem()
    sdp = build([i0, e0, i1, e1])
    sdp.replace_operator_constraints([1, 0], [new_equality, new_inequality])
    assert _snapshot(sdp) == _snapshot(build([new_inequality, e0, i1, new_equality]))


@PROBLEMS
def test_failures_leave_the_relaxation_untouched(problem):
    build, constraints, _, x = problem()
    sdp = build(constraints)
    before = _snapshot(sdp)
    # At level 2, no default generating set fits a degree 6 constraint, which is only found out once the first
    # constraint has been built
    with pytest.raises(ValueError, match="Level isn't large enough"):
        sdp.add_operator_constraints([constraints[0], x**6 >= 0])
    with pytest.raises(IndexError):
        sdp.remove_operator_constraints(equalities=[0], inequalities=[2])
    with pytest.raises(ValueError, match="given twice"):
        sdp.replace_operator_constraints([0, 0], [constraints[1], constraints[1]])
    assert _snapshot(sdp) == before


def test_moment_matrix_id_is_needed_with_several_moment_matrices():
    f0, f1 = generate_noncommutative_variables("F", 2, hermitian=True, moment_matrix_id=0)
    g0, g1 = generate_noncommutative_variables("G", 2, hermitian=True, moment_matrix_id=1)
    constraints = [f0 - f0**2 >= 0, g0 - g0**2 >= 0, g1 - g1**2 >= 0]

    def build(constraints):
        return get_relaxation(
            [f0, f1, g0, g1], 1, f0 * f1 + f1 * f0 + g0 * g1 + g1 * g0, operator_constraints=constraints
        )

    sdp = build(constraints)
    with pytest.raises(ValueError, match="moment_matrix_id must be given"):
        sdp.remove_operator_constraints(inequalities=[0])
    sdp.remove_operator_constraints(inequalities=[0], moment_matrix_id=1)
    assert _snapshot(sdp) == _snapshot(build([f0 - f0**2 >= 0, g1 - g1**2 >= 0]))


def _real_commutative_moments():
    x0, x1 = generate_commutative_variables("x", 2, real=True)
    return (
        lambda moment_constraints: get_relaxation(
            [x0, x1], 2, 2 * x0 * x1, operator_constraints=[x0 - x0**2 >= 0], moment_constraints=moment_constraints
        ),
        [x0 == 0.5, x0 * x1 >= -1, x1**2 == 0.25, x0 + x1 >= 0],
        [x1 == 0.5, x0**2 >= 0],
        # A real-valued relaxation can't take complex constraints
        x0 == 1j,
    )


def _complex_noncommutative_moments():
    a0, a1 = generate_noncommutative_variables("A", 2, hermitian=True)
    return (
        lambda moment_constraints: get_relaxation(
            [a0, a1], 2, (1 + 0j) * (a0 * a1 + a1 * a0), moment_constraints=moment_constraints
        ),
        [a0 == 0.5, a0 * a1 + a1 * a0 >= -1, 1j * (a0 * a1 - a1 * a0) == 0.25, a1 >= 0],
        [a1 == 0.5, a0 * a0 >= 0],
        # The polynomial is anti-Hermitian, so its moment can't be real and non-zero
        a0 * a1 - a1 * a0 == 1,
    )


MOMENT_PROBLEMS = pytest.mark.parametrize("problem", [_real_commutative_moments, _complex_noncommutative_moments])


@MOMENT_PROBLEMS
def test_add_moment_constraints(problem):
    build, (e0, i0, e1, i1), _, _ = problem()
    sdp = build([e0, i0])
    before = _snapshot(sdp)
    sdp.add_moment_constraints([e1, i1])
    expected = build([e0, i0, e1, i1])
    first, second, normalization = expected.moment_equalities
    assert str(sdp.moment_equalities) == str([first, normalization, second])
    assert str(sdp.moment_inequalities) == str(expected.moment_inequalities)
    # Removing them gives back the relaxation as it was
    sdp.remove_moment_constraints(equalities=[2], inequalities=[1])
    assert _snapshot(sdp) == before


@MOMENT_PROBLEMS
def test_remove_moment_constraints(problem):
    build, (e0, i0, e1, i1), _, _ = problem()
    sdp = build([e0, i0, e1, i1])
    sdp.remove_moment_constraints(equalities=[0], inequalities=[1])
    assert _snapshot(sdp) == _snapshot(build([i0, e1]))


@MOMENT_PROBLEMS
def test_replace_moment_constraints_keeps_positions(problem):
    build, (e0, i0, e1, i1), (new_equality, new_inequality), _ = problem()
    sdp = build([e0, i0, e1, i1])
    sdp.replace_moment_constraints([1, 0], [new_equality, new_inequality])
    assert _snapshot(sdp) == _snapshot(build([e0, new_inequality, new_equality, i1]))


@MOMENT_PROBLEMS
def test_moment_failures_leave_the_relaxation_untouched(problem):
    build, constraints, _, invalid = problem()
    sdp = build(constraints)
    before = _snapshot(sdp)
    with pytest.raises(ValueError):
        sdp.add_moment_constraints([constraints[0], invalid])
    with pytest.raises(IndexError):
        sdp.remove_moment_constraints(equalities=[0], inequalities=[2])
    with pytest.raises(ValueError, match="given twice"):
        sdp.replace_moment_constraints([0, 0], [constraints[0], constraints[0]])
    assert _snapshot(sdp) == before
