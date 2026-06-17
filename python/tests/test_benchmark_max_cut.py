import numpy as np
import pytest


@pytest.mark.parametrize("n_vars", [5, 10, 15, 20])
@pytest.mark.parametrize("level", [1, 2])
def test_max_cut_ncpoleon(benchmark, n_vars, level):
    from ncpoleon import generate_commutative_variables, get_relaxation

    variables = generate_commutative_variables("x", n_vars, real=True)
    gen = np.random.default_rng(seed=n_vars)
    random_matrix = gen.random((n_vars, n_vars))
    random_matrix += random_matrix.T
    objective = 0

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            objective += random_matrix[i, j] * (1 - variables[i] * variables[j]) / 2

    substitutions = {v**2: 1 for v in variables}
    benchmark(get_relaxation, variables, level, objective=-objective, substitutions=substitutions)
