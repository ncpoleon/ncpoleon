from math import sqrt

import pytest
from ncpoleon import generate_commutative_variables, generate_noncommutative_variables, get_relaxation, solve

from .test_simple_commutative_problem import _simple_commutative_params
from .test_simple_complex_problem import _simple_complex_params
from .test_simple_noncommutative_problem import _simple_noncommutative_vars
from .utils import SOLVERS, consistency_check


def _relaxations():
    x0, x1 = generate_commutative_variables("x", 2, real=True)
    a0, a1 = generate_noncommutative_variables("A", 2, hermitian=True)
    return {
        "real_commutative": get_relaxation([x0, x1], 2, 2 * x0 * x1, operator_constraints=[x1 - x1**2 >= 0]),
        "complex_commutative": get_relaxation([x0, x1], 2, x0 * x1, operator_constraints=[1j * (x0 - x1) == 0]),
        "real_noncommutative": get_relaxation(
            [a0, a1], 2, a0 * a1 + a1 * a0, operator_constraints=[a0 - a0**2 >= 0], substitutions={a0 * a0: a0}
        ),
        "complex_noncommutative": get_relaxation(
            [a0, a1], 2, a0 * a1 + a1 * a0, operator_constraints=[1j * (a1 * a0 - a0 * a1) == 0]
        ),
    }


def _snapshot(sdp):
    """Everything the relaxation exposes, as comparable strings."""
    return [
        str(sdp.objective),
        str(sdp.generating_sets),
        str(sdp.equalities),
        str(sdp.inequalities),
        str(sdp.moment_equalities),
        str(sdp.moment_inequalities),
        str(sdp.localising_moment_matrices_equalities),
        str({k: mm.as_row_col_data_format() for k, mm in sdp.moment_matrices.items()}),
        str(
            {
                k: [mm.as_row_col_data_format() for mm in mms]
                for k, mms in sdp.localising_moment_matrices_inequalities.items()
            }
        ),
        str(sdp.rewrite(sdp.objective)),
    ]


@pytest.mark.parametrize("fmt", ["postcard", "ron"])
@pytest.mark.parametrize("name", list(_relaxations()))
def test_save_load_round_trip(tmp_path, name, fmt):
    sdp = _relaxations()[name]
    path = tmp_path / f"relaxation.{fmt}"
    sdp.save(path, format=fmt)
    loaded = type(sdp).load(path, format=fmt)
    assert type(loaded) is type(sdp)
    assert _snapshot(loaded) == _snapshot(sdp)


def _solvable_relaxations():
    """One problem per relaxation class, with the known optimum asserted by the solver tests."""
    x0, x1, obj, constraints = _simple_commutative_params()
    y1, y2, nc_obj = _simple_noncommutative_vars()
    z1, z2, complex_obj, complex_constraints = _simple_complex_params()
    return {
        "real_commutative": (get_relaxation([x0, x1], 2, obj, operator_constraints=constraints), "min", 1 - sqrt(2)),
        # A `+0j` coefficient makes the relaxation complex-valued without changing the problem
        "complex_commutative": (
            get_relaxation([x0, x1], 2, (1 + 0j) * obj, operator_constraints=constraints),
            "min",
            1 - sqrt(2),
        ),
        "real_noncommutative": (
            get_relaxation(
                [y1, y2],
                1,
                nc_obj,
                operator_constraints=[y1 - y1**2 >= 0, y2 - y2**2 >= 0],
                substitutions={y2 * y1: y1 * y2},
            ),
            "max",
            1 / 8,
        ),
        "complex_noncommutative": (
            get_relaxation([z1, z2], 2, complex_obj, operator_constraints=complex_constraints),
            "min",
            -2.0,
        ),
    }


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("fmt", ["postcard", "ron"])
@pytest.mark.parametrize("name", list(_solvable_relaxations()))
def test_loaded_relaxation_solves(tmp_path, name, fmt, solver):
    sdp, sense, expected = _solvable_relaxations()[name]
    path = tmp_path / f"relaxation.{fmt}"
    sdp.save(path, format=fmt)
    loaded = type(sdp).load(path, format=fmt)
    sol = solve(loaded, sense, solver=solver)
    assert sol.value == pytest.approx(expected, abs=1e-6)
    consistency_check(loaded, sol, objective_sense=sense, sos_tol=1e-07)


@pytest.mark.parametrize("fmt", ["postcard", "ron"])
def test_load_into_wrong_class_fails(tmp_path, fmt):
    relaxations = _relaxations()
    path = tmp_path / "relaxation"
    relaxations["real_commutative"].save(str(path), format=fmt)
    with pytest.raises(ValueError):
        type(relaxations["complex_noncommutative"]).load(path, format=fmt)


def test_unknown_format(tmp_path):
    sdp = _relaxations()["real_commutative"]
    with pytest.raises(ValueError, match="Unknown format"):
        sdp.save(tmp_path / "relaxation", format="json")
