from math import log2, sqrt

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve

from .utils import SOLVER_SKIPS, reduce_sos_decomposition


def generate_multiple_moment_matrices_parameters():
    res = []

    for solver in ["mosek", "picos-cvxopt"]:
        for use_primal in [True, False]:
            for level, w, expected in [
                (1, 2.0, 0.0),
                (1, 2.2, 0.0),
                (2, 2.0, 0.0),
                (2, 2.2, 1 - log2(1 + sqrt(2 - pow(2.2, 2) / 4))),
            ]:
                marks = [SOLVER_SKIPS[solver]]

                if solver == "mosek" and use_primal and level >= 2:
                    marks.append(
                        pytest.mark.xfail(
                            reason="Solving the primal using the MOSEK Python Fusion API results in a Recursion "
                            "Error because the involved LMI is too large.",
                            raises=RecursionError,
                        )
                    )

                res.append(pytest.param(solver, use_primal, level, w, expected, marks=marks))

    return res


def _multiple_moment_matrices_params(w):
    F, I_0 = generate_noncommutative_variables("F", 4, projector=True, moment_matrix_id=0, return_identity=True)
    G = generate_noncommutative_variables("G", 4, projector=True, moment_matrix_id=0)
    M, I_1 = generate_noncommutative_variables("M", 4, projector=True, moment_matrix_id=1, return_identity=True)
    N = generate_noncommutative_variables("N", 4, projector=True, moment_matrix_id=1)

    substitutions = {}
    for g in G:
        for f in F:
            substitutions[g * f] = f * g
    for n in N:
        for m in M:
            substitutions[n * m] = m * n

    operator_constraints = [
        F[0] + F[2] == I_0,
        F[1] + F[3] == I_0,
        G[0] + G[2] == I_0,
        G[1] + G[3] == I_0,
        M[0] + M[2] == I_1,
        M[1] + M[3] == I_1,
        N[0] + N[2] == I_1,
        N[1] + N[3] == I_1,
    ]

    F_0 = F[0] - F[2]
    F_1 = F[1] - F[3]
    G_0 = G[0] - G[2]
    G_1 = G[1] - G[3]
    M_0 = M[0] - M[2]
    M_1 = M[1] - M[3]
    N_0 = N[0] - N[2]
    N_1 = N[1] - N[3]

    K_1 = F_0 * (G_0 + G_1) + F_1 * (G_0 - G_1)
    K_2 = M_0 * (N_0 + N_1) + M_1 * (N_0 - N_1)

    moment_constraints = [K_1 + K_2 == w]
    normalization_constraints = [I_0 + I_1 == 1]
    objective = F[0] + M[2]

    # The operators of each moment matrix, ordered so that the family the other one commutes past
    # comes first, together with the identity of that moment matrix
    moment_matrices_operators = [(F + G, I_0), (M + N, I_1)]

    return (
        F + G + M + N,
        objective,
        substitutions,
        operator_constraints,
        moment_constraints,
        normalization_constraints,
        moment_matrices_operators,
    )


def _generating_set(level, operators, identity):
    """Rebuild the generating set that ``get_relaxation`` builds at ``level`` for one moment matrix.

    ``operators`` is the concatenation of the two commuting families of operators of that moment
    matrix, the family the other one commutes past coming first. Only the levels 0, 1 and 2 that the
    tests below need are supported.
    """
    monomials = [identity]

    if level >= 1:
        monomials.extend(operators)

    if level >= 2:
        first_family_size = len(operators) // 2

        for index_left, left in enumerate(operators):
            for index_right, right in enumerate(operators):
                # The operators are projectors, hence the squares reduce to a monomial of length 1,
                # and the second family commutes past the first one, hence `right * left` reduces to
                # the already generated `left * right`
                if (index_left == index_right) or (index_left >= first_family_size > index_right):
                    continue

                monomials.append(left * right)

    return monomials


@pytest.mark.parametrize("level", [1, 2])
def test_multiple_moment_matrices_relaxation(benchmark, level):
    # TODO: write docstring about the problem and change the name, it's about CHSH
    variables, objective, substitutions, operator_constraints, moment_constraints, normalization_constraints, _ = (
        _multiple_moment_matrices_params(2.0)
    )
    benchmark(
        get_relaxation,
        variables,
        level,
        objective,
        substitutions=substitutions,
        operator_constraints=operator_constraints,
        moment_constraints=moment_constraints,
        normalization_constraints=normalization_constraints,
    )


@pytest.mark.parametrize("solver, use_primal, level, w, expected", generate_multiple_moment_matrices_parameters())
@pytest.mark.walltime
def test_multiple_moment_matrices_solve(benchmark, solver, use_primal, level, w, expected):
    # TODO: write docstring about the problem and change the name, it's about CHSH
    variables, objective, substitutions, operator_constraints, moment_constraints, normalization_constraints, _ = (
        _multiple_moment_matrices_params(w)
    )
    sdp = get_relaxation(
        variables,
        level,
        objective,
        substitutions=substitutions,
        operator_constraints=operator_constraints,
        moment_constraints=moment_constraints,
        normalization_constraints=normalization_constraints,
    )
    sol = benchmark(solve, sdp, "max", force_primal=use_primal, solver=solver)
    assert -log2(sol.value) == pytest.approx(expected, abs=1e-6)
    sos_decompositions = sol.get_sos_decomposition_by_mm_id()
    reduced_0 = reduce_sos_decomposition(sos_decompositions[0])
    reduced_1 = reduce_sos_decomposition(sos_decompositions[1])
    assert sdp.rewrite(reduced_0 + reduced_1 + objective).is_zero(1e-7)


@pytest.mark.parametrize("solver, use_primal, level, w, expected", generate_multiple_moment_matrices_parameters())
@pytest.mark.walltime
def test_multiple_moment_matrices_with_extra_monomials(benchmark, solver, use_primal, level, w, expected):
    """Solve the problem at level -1, the generating sets being given rather than generated.

    The generating sets that are handed over are the ones that ``get_relaxation`` would have built
    at ``level``, so that the relaxation, and hence the optimal value, are the ones of ``level``.
    """
    (
        variables,
        objective,
        substitutions,
        operator_constraints,
        moment_constraints,
        normalization_constraints,
        moment_matrices_operators,
    ) = _multiple_moment_matrices_params(w)

    extra_monomials = [
        monomial
        for operators, identity in moment_matrices_operators
        for monomial in _generating_set(level, operators, identity)
    ]

    # The operator constraints all are of degree 1, so that their localising moment matrices are
    # indexed by the generating set of the previous level. The first four constraints belong to the
    # first moment matrix, the last four to the second one
    localising_generating_sets = [
        _generating_set(level - 1, operators, identity) for operators, identity in moment_matrices_operators
    ]
    operator_constraints = [
        (constraint, localising_generating_sets[index // 4]) for index, constraint in enumerate(operator_constraints)
    ]

    sdp = benchmark(
        get_relaxation,
        [],
        level=-1,
        objective=objective,
        substitutions=substitutions,
        operator_constraints=operator_constraints,
        moment_constraints=moment_constraints,
        normalization_constraints=normalization_constraints,
        extra_monomials=extra_monomials,
    )
    sol = solve(sdp, "max", verbosity=0, force_primal=use_primal, solver=solver)
    assert -log2(sol.value) == pytest.approx(expected, abs=1e-6)
    sos_decompositions = sol.get_sos_decomposition_by_mm_id()
    reduced_0 = reduce_sos_decomposition(sos_decompositions[0])
    reduced_1 = reduce_sos_decomposition(sos_decompositions[1])
    assert sdp.rewrite(reduced_0 + reduced_1 + objective).is_zero(1e-7)
