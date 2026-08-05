import numpy as np
import pytest


def _max_cut_params(n_vars):
    """Build the max-cut objective and substitutions over ``n_vars`` real commutative variables."""
    from ncpoleon import generate_commutative_variables

    variables = generate_commutative_variables("x", n_vars, real=True)
    gen = np.random.default_rng(seed=n_vars)
    random_matrix = gen.random((n_vars, n_vars))
    random_matrix += random_matrix.T
    objective = 0

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            objective += random_matrix[i, j] * (1 - variables[i] * variables[j]) / 2

    return variables, -objective, {v**2: 1 for v in variables}


@pytest.mark.parametrize("n_vars", [5, 10, 15, 20])
@pytest.mark.parametrize("level", [1, 2])
def test_max_cut_ncpoleon(benchmark, n_vars, level):
    from ncpoleon import get_relaxation

    variables, objective, substitutions = _max_cut_params(n_vars)
    benchmark(get_relaxation, variables, level, objective=objective, substitutions=substitutions)


@pytest.mark.walltime
def test_max_cut_relaxation_walltime(benchmark):
    """Measure ``get_relaxation`` in walltime mode.

    The other walltime benchmarks all measure ``solve``, and therefore the third-party solvers
    rather than our own code. Instruction counts are blind to cache misses and memory bandwidth,
    which is where the moment matrix fill actually spends its time, so the hot path needs at
    least one wall-clock benchmark of its own.

    ``n_vars=25`` takes roughly 0.67s, comfortably above the measurement noise floor while
    keeping the job short.
    """
    from ncpoleon import get_relaxation

    variables, objective, substitutions = _max_cut_params(25)
    benchmark(get_relaxation, variables, 2, objective=objective, substitutions=substitutions)
