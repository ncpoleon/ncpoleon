import logging

import pytest
from ncpoleon import generate_commutative_variables, generate_noncommutative_variables, get_relaxation
from ncpoleon.logging import FORMAT, set_logging_format, set_notebook, set_verbosity_level
from ncpoleon.relaxations import Canonicality, Realness


def _invalid_relaxation_cases():
    x, y = generate_commutative_variables("x", 2, real=True)
    z = generate_commutative_variables("z", 1, real=True, moment_matrix_id=1)[0]
    a, b = generate_noncommutative_variables("a", 2)
    h0, h1 = generate_noncommutative_variables("h", 2, hermitian=True)
    c = generate_noncommutative_variables("c", 1, hermitian=True, moment_matrix_id=1)[0]

    return [
        pytest.param([x], 1, "foo", {}, ValueError, "objective", id="objective"),
        pytest.param(["foo"], 1, x, {}, ValueError, "variable at index 0", id="variable"),
        pytest.param(
            [x], 1, x, {"extra_monomials": ["foo"]}, ValueError, "extra monomial at index 0", id="extra_monomial"
        ),
        pytest.param(
            [x], 1, x, {"operator_constraints": ["foo"]}, ValueError, "operator constraint", id="operator_constraint"
        ),
        pytest.param(
            [x],
            1,
            x,
            {"operator_constraints": [(x >= 0, ["foo"])]},
            ValueError,
            "generating set",
            id="generating_set",
        ),
        pytest.param([x], 1, x, {"moment_constraints": ["foo"]}, ValueError, "moment constraint", id="moment"),
        pytest.param(
            [x],
            1,
            x,
            {"normalization_constraints": ["foo"]},
            ValueError,
            "normalization constraint",
            id="normalization",
        ),
        pytest.param([x], 1, x, {"substitutions": {"foo": x}}, ValueError, "key at index 0", id="substitution_key"),
        pytest.param([x], 1, x, {"substitutions": {x: "foo"}}, ValueError, "substitute", id="substitution_value"),
        pytest.param([], 1, 1.0, {}, ValueError, "Variables must be provided", id="no_variables"),
        pytest.param([x, a], 1, x, {}, NotImplementedError, "Hybrid", id="hybrid"),
        pytest.param([x], -2, x, {}, ValueError, "level must be larger", id="level"),
        pytest.param([a, b], 1, a * b, {}, ValueError, "must be hermitian", id="non_hermitian_objective"),
        # A rule given without its adjoint counterpart
        pytest.param(
            [a, b],
            2,
            a * b + (a * b).adjoint(),
            {"substitutions": {b * a: a * b}},
            ValueError,
            "not adjoint-stable",
            id="adjoint_unstable_substitutions",
        ),
        pytest.param(
            [x, y], 1, x, {"moment_constraints": [x >= y]}, ValueError, "both sides are polynomials", id="poly_vs_poly"
        ),
        pytest.param(
            [x], 1, x, {"moment_constraints": [x >= 1j]}, ValueError, "non-zero imaginary part", id="complex_bound"
        ),
        pytest.param(
            [x, z],
            1,
            x + z,
            {"substitutions": {x * x: z}},
            ValueError,
            "Cannot substitute",
            id="commutative_substitution_across_moment_matrices",
        ),
        pytest.param(
            [h0, h1, c],
            1,
            h0 + c,
            {"substitutions": {h0 * h1: c, h1 * h0: c}},
            ValueError,
            "Cannot substitute",
            id="noncommutative_substitution_across_moment_matrices",
        ),
    ]


@pytest.mark.parametrize("variables, level, objective, kwargs, error, match", _invalid_relaxation_cases())
def test_invalid_relaxation(variables, level, objective, kwargs, error, match):
    with pytest.raises(error, match=match):
        get_relaxation(variables, level, objective, **kwargs)


def test_scalar_on_the_left_of_a_constraint():
    x, y = generate_commutative_variables("x", 2, real=True)

    flipped = get_relaxation([x, y], 1, x * y, operator_constraints=[x <= 1.0], moment_constraints=[x * y <= 0.5])
    expected = get_relaxation(
        [x, y], 1, x * y, operator_constraints=[1.0 - x >= 0], moment_constraints=[-(x * y) >= -0.5]
    )

    assert str(flipped.inequalities) == str(expected.inequalities)
    assert str(flipped.moment_inequalities) == str(expected.moment_inequalities)


@pytest.mark.parametrize("real", [True, False])
@pytest.mark.parametrize("commutative", [True, False])
def test_relaxation_lookups(real, commutative):
    if commutative:
        (x, y), identity = generate_commutative_variables("x", 2, real=True, return_identity=True)
    else:
        (x, y), identity = generate_noncommutative_variables("x", 2, hermitian=True, return_identity=True)
    objective = x * y + y * x if real else x * y + y * x + 0j
    sdp = get_relaxation([x, y], 1, objective)
    moment_matrix = sdp.moment_matrices[0]

    assert x in moment_matrix
    assert x * y in moment_matrix
    assert x**3 not in moment_matrix
    assert moment_matrix.get_canonical(x)[1] == Canonicality.Canonical
    assert moment_matrix.get_canonical(x)[2] == Realness.Real

    assert str(sdp.rewrite(x)) == str(x)
    assert str(sdp.rewrite(x * y)) == str(x * y)
    assert str(sdp.rewrite(1.5)) == str(1.5 * identity)
    if not real:
        assert str(sdp.rewrite(1.5 + 1j)) == str((1.5 + 1j) * identity)

    with pytest.raises(KeyError, match="Couldn't find monomial"):
        sdp.change_variables(sdp.objective, {})


def _two_moment_matrices():
    a = generate_noncommutative_variables("a", 2, hermitian=True, moment_matrix_id=0)
    b = generate_noncommutative_variables("b", 2, hermitian=True, moment_matrix_id=1)
    objective = a[0] * a[1] + a[1] * a[0] + b[0] * b[1] + b[1] * b[0]
    # One localizing matrix of each kind, so that each of their progress bars is drawn
    operator_constraints = [a[0] - a[0] ** 2 >= 0, b[0] - b[0] ** 2 == 0]
    return a + b, objective, operator_constraints


@pytest.mark.parametrize("verbosity", [1, 2, 3])
def test_verbosity(verbosity):
    """Logging and progress bars, over two moment matrices so that every bar is drawn."""
    variables, objective, operator_constraints = _two_moment_matrices()
    expected = get_relaxation(variables, 2, objective, operator_constraints=operator_constraints)

    set_notebook(False)
    try:
        sdp = get_relaxation(variables, 2, objective, operator_constraints=operator_constraints, verbosity=verbosity)
        assert logging.getLogger("ncpoleon").level == {1: logging.INFO, 2: logging.DEBUG, 3: 5}[verbosity]
    finally:
        set_verbosity_level(0)

    assert str(sdp.generating_sets) == str(expected.generating_sets)
    assert str(sdp.equalities) == str(expected.equalities)
    assert str(sdp.inequalities) == str(expected.inequalities)
    assert logging.getLogger("ncpoleon").level == logging.WARNING


def test_rewriting_a_scalar_needs_a_unique_moment_matrix():
    variables, objective, _ = _two_moment_matrices()

    for objective, scalars in [(objective, [1.0, 1.5]), (objective + 0j, [1.5 + 1j])]:
        sdp = get_relaxation(variables, 1, objective)
        for scalar in scalars:
            with pytest.raises(TypeError, match="unique moment matrix"):
                sdp.rewrite(scalar)


@pytest.mark.parametrize("kind", ["operator", "complex_scalar", "complex_polynomial"])
def test_objective_kinds(kind):
    a = generate_noncommutative_variables("a", 1, hermitian=True)[0]
    objective = {"operator": a, "complex_scalar": 2 + 0j, "complex_polynomial": a + 0j}[kind]

    sdp = get_relaxation([a], 1, objective)

    assert sdp.is_real is (kind == "operator")
    assert (objective - sdp.objective).is_zero()


def test_normalization_inequality_replaces_the_default_normalization():
    (x,), identity = generate_commutative_variables("x", 1, real=True, return_identity=True)

    default = get_relaxation([x], 1, x)
    custom = get_relaxation([x], 1, x, normalization_constraints=[identity >= 1])

    assert (len(default.moment_equalities), len(default.moment_inequalities)) == (1, 0)
    assert (len(custom.moment_equalities), len(custom.moment_inequalities)) == (0, 1)


def test_uniqueness_check_without_length():
    a = generate_noncommutative_variables("a", 2, hermitian=True)
    objective = a[0] * a[1] + a[1] * a[0]
    # Shortens monomials, so the length shortcut and the full scan over previous levels must agree
    substitutions = {a[0] * a[0]: a[0], a[1] * a[1]: a[1]}

    with_length = get_relaxation(a, 2, objective, substitutions=substitutions)
    without_length = get_relaxation(a, 2, objective, substitutions=substitutions, check_uniqueness_with_length=False)

    assert str(without_length.generating_sets) == str(with_length.generating_sets)


def test_invalid_verbosity():
    with pytest.raises(ValueError, match="verbosity must be between 0 and 3"):
        set_verbosity_level(4)


def test_set_logging_format():
    handler = logging.getLogger("ncpoleon").handlers[0]
    try:
        set_logging_format("%(message)s")
        assert handler.formatter._fmt == "%(message)s"
    finally:
        set_logging_format(FORMAT)
