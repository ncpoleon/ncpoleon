import importlib
import sys
from contextlib import nullcontext

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve
from ncpoleon.export import to_mosek, to_picos
from ncpoleon.logging import set_verbosity_level
from ncpoleon.solve import MosekSolution
from ncpoleon.solve.utils import automatic_solver_detection
from ncpoleon.utils import is_mosek_available

from .utils import (
    MOSEK_AVAILABLE,
    MOSEK_SKIP_REASON,
    SOLVER_SKIPS,
    SOLVERS,
    consistency_check,
    simple_commutative_params,
)

# `ncpoleon.solve.solve` and friends are shadowed by the functions of the same name, hence `import_module`
utils_module = importlib.import_module("ncpoleon.utils")
solve_module = importlib.import_module("ncpoleon.solve.solve")
solve_utils_module = importlib.import_module("ncpoleon.solve.utils")
to_mosek_module = importlib.import_module("ncpoleon.export.to_mosek")
to_picos_module = importlib.import_module("ncpoleon.export.to_picos")


def _relaxation():
    x0, x1, obj, operator_constraints = simple_commutative_params()
    return get_relaxation([x0, x1], 1, obj, operator_constraints=operator_constraints)


EXPECTED = -0.5


def test_is_mosek_available_without_mosek(monkeypatch):
    monkeypatch.setattr(utils_module, "find_spec", lambda name: None)
    assert not is_mosek_available()


def test_is_mosek_available_with_broken_mosek(monkeypatch):
    monkeypatch.setattr(utils_module, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "mosek", None)
    assert not is_mosek_available()


def test_is_mosek_available_without_license(monkeypatch, caplog):
    # Never let this reach a real license checkout, which can block for a minute when no license is found
    mosek = pytest.importorskip("mosek")

    class Env:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def checkoutlicense(self, feature):
            raise mosek.Error(mosek.rescode.err_license, "No license.")

    monkeypatch.setattr(mosek, "Env", Env)
    assert not is_mosek_available()
    assert "no valid license" in caplog.text


@pytest.mark.parametrize(
    "mosek, picos, expected", [(True, True, "mosek"), (False, True, "picos"), (False, False, None)]
)
def test_automatic_solver_detection(monkeypatch, mosek, picos, expected):
    monkeypatch.setattr(solve_utils_module, "is_mosek_available", lambda: mosek)
    monkeypatch.setattr(solve_utils_module, "find_spec", lambda name: object() if picos else None)

    if expected is None:
        with pytest.raises(ImportError, match="No solver has been found"):
            automatic_solver_detection()
    else:
        assert automatic_solver_detection() == expected


@SOLVER_SKIPS["picos-cvxopt"]
def test_solve_auto_with_picos(monkeypatch):
    # Left to itself, PICOS would pick MOSEK whenever it is installed, even without a license
    monkeypatch.setattr(pytest.importorskip("picos").settings, "SOLVER_WHITELIST", ["cvxopt"])
    monkeypatch.setattr(solve_module, "automatic_solver_detection", lambda: "picos")
    assert solve(_relaxation(), "min").value == pytest.approx(EXPECTED, abs=1e-6)


def test_solve_unknown_solver():
    with pytest.raises(ValueError, match="isn't a valid solver"):
        solve(_relaxation(), "min", solver="foo")


@pytest.mark.parametrize(
    "export, module, available",
    [(to_mosek, to_mosek_module, "_mosek_available"), (to_picos, to_picos_module, "_picos_available")],
)
def test_export_errors(monkeypatch, export, module, available):
    sdp = _relaxation()

    if getattr(module, available):
        with pytest.raises(ValueError, match="objective directions"):
            export(sdp, "up", primal=False)

    monkeypatch.setattr(module, available, False)
    with pytest.raises(ImportError, match="is required"):
        export(sdp, "min", primal=False)


@pytest.mark.skipif(not MOSEK_AVAILABLE, reason=MOSEK_SKIP_REASON)
def test_mosek_verbosity_and_parameters(capfd):
    sdp = _relaxation()
    try:
        sol = solve(sdp, "min", solver="mosek", verbosity=1, intpntCoTolRelGap=1e-9)
    finally:
        set_verbosity_level(0)
    assert sol.value == pytest.approx(EXPECTED, abs=1e-6)
    assert capfd.readouterr().out


@pytest.mark.skipif(not MOSEK_AVAILABLE, reason=MOSEK_SKIP_REASON)
def test_mosek_solution_errors():
    sdp = _relaxation()
    model = to_mosek(sdp, "min", primal=False)

    with pytest.raises(ValueError, match="objective_sense"):
        MosekSolution(sdp, model, primal=False, objective_sense="up")

    sol = MosekSolution(sdp, model, primal=False, objective_sense="min")
    with pytest.raises(KeyError, match="no constraint named foo"):
        sol._constraint("foo")
    with pytest.raises(KeyError, match="no variable named foo"):
        sol._variable("foo")


SHORTCUTS = [
    ("moment_matrix", "moment_matrix_by_mm_id"),
    ("moment_matrix_multiplier", "moment_matrix_multiplier_by_mm_id"),
    (
        "localizing_matrices_hermitian_equality_multipliers",
        "localizing_matrices_hermitian_equality_multipliers_by_mm_id",
    ),
    (
        "localizing_matrices_antihermitian_equality_multipliers",
        "localizing_matrices_antihermitian_equality_multipliers_by_mm_id",
    ),
    (
        "localizing_matrices_nonhermitian_equality_multipliers",
        "localizing_matrices_nonhermitian_equality_multipliers_by_mm_id",
    ),
    ("localizing_matrices_inequality", "localizing_matrices_inequality_by_mm_id"),
    ("localizing_matrices_inequality_multipliers", "localizing_matrices_inequality_multipliers_by_mm_id"),
    ("get_sos_decomposition", "get_sos_decomposition_by_mm_id"),
]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("moment_matrices", [1, 2])
def test_single_moment_matrix_shortcuts(solver, moment_matrices):
    """Each shortcut is the moment matrix 0 entry of its `_by_mm_id` counterpart, warning when there are others."""
    variables, objective, operator_constraints, moment_constraints = [], 0, [], []
    for mm_id in range(moment_matrices):
        x1, x2 = generate_noncommutative_variables("xy"[mm_id], 2, hermitian=True, moment_matrix_id=mm_id)
        variables += [x1, x2]
        objective += x1 * x2 + x2 * x1
        operator_constraints += [
            x1 - x1**2 >= 0,
            x2 - x2**2 == 0,
            x2 * x1 - x1 * x2 == 0,
            x1 - x1**2 + x2 * x1 - x1 * x2 == 0,
        ]
        moment_constraints += [x1 >= -1, x2 >= -1]
    sdp = get_relaxation(
        variables, 2, objective, operator_constraints=operator_constraints, moment_constraints=moment_constraints
    )
    sol = solve(sdp, "min", solver=solver)
    # A `delta` takes the chopping path of the SoS decomposition
    consistency_check(sdp, sol, objective_sense="min", sos_tol=1e-6, delta=1e-12)

    for shortcut, by_mm_id in SHORTCUTS:
        expected = getattr(sol, by_mm_id)
        expected = expected()[0] if callable(expected) else expected[0]
        with pytest.warns(UserWarning, match="multiple moment matrices") if moment_matrices > 1 else nullcontext():
            actual = getattr(sol, shortcut)
            actual = actual() if callable(actual) else actual
        assert str(actual) == str(expected)
