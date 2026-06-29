from math import log2, sqrt

import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation, solve
from ncpoleon.utils import is_mosek_available

from .utils import reduce_sos_decomposition


def generate_multiple_moment_matrices_parameters():
    for solver in ["mosek", "picos-cvxopt"]:
        for use_primal in [True, False]:
            for level, w, expected in [
                (1, 2.0, 0.0),
                (1, 2.2, 0.0),
                (2, 2.0, 0.0),
                (2, 2.2, 1 - log2(1 + sqrt(2 - pow(2.2, 2) / 4))),
            ]:
                marks = []

                if solver == "mosek":
                    marks.append(
                        pytest.mark.skipif(
                            not is_mosek_available(),
                            reason="Mosek is not installed or a Mosek license is not available.",
                        )
                    )

                    if use_primal and level >= 2:
                        marks.append(
                            pytest.mark.xfail(
                                reason="Solving the primal using the MOSEK Python Fusion API results in a Recursion "
                                "Error because the involved LMI is too large.",
                                raises=RecursionError,
                            )
                        )

                yield pytest.param(solver, use_primal, level, w, expected, marks=marks)


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

    return F + G + M + N, objective, substitutions, operator_constraints, moment_constraints, normalization_constraints


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("w", [2.0, 2.2])
def test_multiple_moment_matrices_relaxation(benchmark, level, w):
    # TODO: write docstring about the problem and change the name, it's about CHSH
    variables, objective, substitutions, operator_constraints, moment_constraints, normalization_constraints = (
        _multiple_moment_matrices_params(w)
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


# TODO: benchmark this once the Codspeed action is setup
@pytest.mark.parametrize("solver, use_primal, level, w, expected", generate_multiple_moment_matrices_parameters())
def test_multiple_moment_matrices(solver, use_primal, level, w, expected):
    # TODO: write docstring about the problem and change the name, it's about CHSH
    variables, objective, substitutions, operator_constraints, moment_constraints, normalization_constraints = (
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
    sol = solve(sdp, "max", force_primal=use_primal, solver=solver)
    assert -log2(sol.value) == pytest.approx(expected, abs=1e-6)
    sos_decompositions = sol.get_sos_decomposition_by_mm_id()
    reduced_0 = reduce_sos_decomposition(sos_decompositions[0])
    reduced_1 = reduce_sos_decomposition(sos_decompositions[1])
    assert sdp.rewrite(reduced_0 + reduced_1 + objective).is_zero(1e-7)
