"""Every arithmetic operator between every kind of polynomial element, checked against a pure-Python reference."""

import operator
from collections import Counter, defaultdict
from itertools import product

import pytest
from ncpoleon import generate_commutative_variables, generate_noncommutative_variables

GENERATORS = {"commutative": generate_commutative_variables, "noncommutative": generate_noncommutative_variables}
KINDS = ["operator", "monomial", "identity", "real_poly", "complex_poly", "float", "complex"]
SCALAR_KINDS = {"float", "complex"}
NON_SCALAR_KINDS = [kind for kind in KINDS if kind not in SCALAR_KINDS]
COMPLEX_KINDS = {"complex_poly", "complex"}
OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]
# Scalar-scalar pairs are plain Python arithmetic
PAIRS = [(lhs, rhs) for lhs, rhs in product(KINDS, KINDS) if not {lhs, rhs} <= SCALAR_KINDS]


def _elements(family, moment_matrix_id=0):
    (x, y), identity = GENERATORS[family]("x", 2, moment_matrix_id=moment_matrix_id, return_identity=True)
    return {
        "operator": x,
        "monomial": x * y,
        "identity": identity,
        "real_poly": x + 2.0,
        "complex_poly": y + 1j,
        "float": 1.5,
        "complex": 2 + 1j,
    }


def _reference_terms(family):
    """The terms of `_elements` in moment matrix 0, each monomial written as the word of its operators' names."""
    x, y = (str(variable) for variable in GENERATORS[family]("x", 2))
    return {
        "operator": [((x,), 1.0)],
        "monomial": [((x, y), 1.0)],
        "identity": [((), 1.0)],
        "real_poly": [((x,), 1.0), ((), 2.0)],
        "complex_poly": [((y,), 1.0), ((), 1j)],
        "float": [((), 1.5)],
        "complex": [((), 2 + 1j)],
    }


def _render(family, word):
    """How the library displays the moment matrix 0 monomial that `word` stands for."""
    if family == "noncommutative":
        return "".join(word) or "I_0"
    powers = sorted(Counter(word).items())
    return "".join(name if power == 1 else f"{name}^{power}" for name, power in powers) or "1"


def _accumulate(terms):
    res = defaultdict(complex)
    for key, coefficient in terms:
        res[key] += coefficient
    return {key: coefficient for key, coefficient in res.items() if coefficient != 0}


# Keyed by moment matrix and `str`, as monomials' `__eq__` builds a (truthy) constraint and can't key a dict
def _as_dict(value):
    terms = value.as_dict().items() if hasattr(value, "as_dict") else [(value, 1.0)]
    return _accumulate(((monomial.moment_matrix_id, str(monomial)), coefficient) for monomial, coefficient in terms)


def _reference(family, op, lhs_kind, rhs_kind):
    terms = _reference_terms(family)
    left, right = terms[lhs_kind], terms[rhs_kind]
    if op in (operator.mul, operator.truediv):
        # Division is only ever by a scalar, whose word is empty
        words = [(w1 + w2, op(c1, c2)) for (w1, c1), (w2, c2) in product(left, right)]
    else:
        sign = 1 if op is operator.add else -1
        words = left + [(w, sign * c) for w, c in right]
    return _accumulate(((0, _render(family, word)), coefficient) for word, coefficient in words)


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("lhs_kind, rhs_kind", PAIRS)
def test_binary_operator(family, op, lhs_kind, rhs_kind):
    elements = _elements(family)
    lhs, rhs = elements[lhs_kind], elements[rhs_kind]

    if op is operator.truediv and rhs_kind not in SCALAR_KINDS:
        with pytest.raises(TypeError):
            op(lhs, rhs)
        return

    res = op(lhs, rhs)

    assert _as_dict(res) == pytest.approx(_reference(family, op, lhs_kind, rhs_kind))
    is_complex = bool({lhs_kind, rhs_kind} & COMPLEX_KINDS)
    if hasattr(res, "is_real"):
        assert res.is_real is not is_complex
    else:
        assert not is_complex


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("op", OPERATORS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("kind", NON_SCALAR_KINDS)
def test_unsupported_operand(family, op, kind):
    element = _elements(family)[kind]

    with pytest.raises(TypeError):
        op(element, None)
    with pytest.raises(TypeError):
        op(None, element)


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("kind", NON_SCALAR_KINDS)
def test_unary_methods(family, kind):
    element = _elements(family)[kind]
    expected = _as_dict(element)

    assert _as_dict(-element) == {key: -coefficient for key, coefficient in expected.items()}
    assert _as_dict(element**2) == pytest.approx(_reference(family, operator.mul, kind, kind))
    assert _as_dict(element.adjoint().adjoint()) == expected
    assert repr(element) == str(element)

    if kind in {"operator", "monomial", "identity"}:
        assert element.moment_matrix_id == 0
        assert hash(element) == hash(element.adjoint().adjoint())
    if kind in {"monomial", "identity"}:
        assert len(element) == (2 if kind == "monomial" else 0)
    if kind in {"real_poly", "complex_poly"}:
        assert element.degree() == 1
        assert element.is_real is (kind == "real_poly")


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("kind", NON_SCALAR_KINDS)
def test_comparisons_build_constraints(family, kind):
    elements = _elements(family)
    element = elements[kind]

    greater = element >= 1.0
    assert greater.is_inequality and not greater.is_equality
    assert _as_dict(greater.lhs) == _as_dict(element)
    assert greater.rhs == 1.0

    # The element's own `__le__` swaps the operands, so the scalar ends up on the left
    lower = element <= 1.0
    assert lower.is_inequality
    assert lower.lhs == 1.0
    assert _as_dict(lower.rhs) == _as_dict(element)
    assert str(lower) == repr(lower)
    assert str(lower).endswith(f" >= {lower.rhs})")

    equal = element == elements["operator"]
    assert equal.is_equality and not equal.is_inequality
    assert str(equal).startswith("Constraint(")

    with pytest.raises(TypeError):
        element >= "a"  # noqa: B015


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("lhs_kind", NON_SCALAR_KINDS)
@pytest.mark.parametrize("rhs_kind", NON_SCALAR_KINDS)
def test_multiplying_different_moment_matrices_fails(family, lhs_kind, rhs_kind):
    lhs = _elements(family, moment_matrix_id=0)[lhs_kind]
    rhs = _elements(family, moment_matrix_id=1)[rhs_kind]

    with pytest.raises(ValueError):
        lhs * rhs


@pytest.mark.parametrize("family", GENERATORS)
def test_polynomial_spanning_moment_matrices(family):
    x = GENERATORS[family]("x", 1, moment_matrix_id=0)[0]
    y = GENERATORS[family]("y", 1, moment_matrix_id=1)[0]
    poly = x + y

    for scalar in [1.5, 2 + 1j]:
        with pytest.raises(ValueError, match="ambiguous"):
            poly + scalar
        with pytest.raises(ValueError, match="ambiguous"):
            scalar - poly
    for other in [poly, x, x * x]:
        with pytest.raises(ValueError, match="different moment matrix indices"):
            poly * other
    with pytest.raises(ValueError):
        poly**2


@pytest.mark.parametrize("family", GENERATORS)
def test_zero_polynomial(family, caplog):
    # Built in moment matrix 1, so that the fallback to moment matrix 0 below is observable
    elements = _elements(family, moment_matrix_id=1)

    for kind in ["real_poly", "complex_poly"]:
        zero = elements[kind] - elements[kind]
        assert zero.is_zero()
        assert str(zero) == "0"
        assert (zero**2).is_zero()
        # A zero polynomial has no moment matrix to take the identity from, so it falls back to moment matrix 0
        assert _as_dict(zero + 1.5) == {(0, _render(family, ())): 1.5}
    assert "was added to a zero polynomial" in caplog.text


@pytest.mark.parametrize(
    "family, expected",
    [
        ("commutative", "1+2i + 2 * x_(0) - 3i * x_(0)x_(1) + (1+2i) * x_(1)"),
        ("noncommutative", "(1+2i) * I + 2 * x(0) - 3i * x(0)x(1) + (1+2i) * x(1)"),
    ],
)
def test_complex_polynomial_display(family, expected):
    (x, y), identity = GENERATORS[family]("x", 2, return_identity=True)
    assert str((1 + 2j) * identity + 2 * x + (1 + 2j) * y - 3j * (x * y)) == expected


@pytest.mark.parametrize("family", GENERATORS)
def test_projectors_are_idempotent(family):
    p, q = GENERATORS[family]("p", 2, projector=True)

    assert str(p * p) == str(p**3) == str(p)
    assert str((p * q) * q) == str(p * (p * q)) == str(p * q)
    assert str((p + q) * q) == str(p * q + q)


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("kind", ["real_poly", "complex_poly"])
def test_change_variables(family, kind):
    poly = _elements(family)[kind]
    terms = list(poly.as_dict().items())

    assert poly.change_variables({monomial: 2.0 for monomial, _ in terms}) == pytest.approx(
        sum(2.0 * coefficient for _, coefficient in terms)
    )
    # Every way the partial results can be missing
    for kept in [[0], [1], []]:
        with pytest.raises(ValueError, match="Couldn't find monomial"):
            poly.change_variables({terms[index][0]: 2.0 for index in kept})
