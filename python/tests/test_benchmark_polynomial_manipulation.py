from itertools import permutations
from math import sqrt

import pytest
from ncpoleon.polynomials.commutative_polynomials import generate_commutative_variables


def _build_matrix_for_povm(variables):
    """Build the matrix of which we want to compute the determinant.

    There must be (d**2 - 1) * (d - 1) * 2 variables: d**2 vectors, the first one
    is only on |0>, the others have their first entry set to 1/d and the d-1 others
    to a+ib. As such, this function accepts a list of list of tuples of variables.
    """
    d = len(variables[0]) + 1
    res = [[1 / d**2] + [0.0] * (d**2 - 1)]

    for vars in variables:
        to_append = [1 / d**2]

        # Adding the diagonal
        for i in range(d - 1):
            to_append.append(vars[i][0] * vars[i][0] + vars[i][1] * vars[i][1])

        # First line
        for i in range(d - 1):
            to_append.append(vars[i][0] * (1 / d))  # Real part
            to_append.append(vars[i][1] * (1 / d))  # Real part

        # Other lines
        for i in range(d - 1):
            for j in range(i + 1, d - 1):
                a_i, b_i = vars[i]
                a_j, b_j = vars[j]
                to_append.append(a_i * a_j + b_i * b_j)
                to_append.append(a_j * b_i - a_i * b_j)

        res.append(to_append)

    return res


def _build_matrix(variables):
    res = []
    d = round(sqrt(len(variables)))

    for i in range(d):
        res.append([])
        for j in range(d):
            res[-1].append(variables[d * i + j])

    return res


def generate_ncpoleon_variables_for_povm(d):
    variables_real = generate_commutative_variables("x", (d - 1) * (d**2 - 1))
    variables_imag = generate_commutative_variables("y", (d - 1) * (d**2 - 1))
    return [
        [(variables_real[j + i * (d - 1)], variables_imag[j + i * (d - 1)]) for j in range(d - 1)]
        for i in range(d**2 - 1)
    ]


def generate_ncpoleon_variables(d):
    return generate_commutative_variables("x", d**2)


def compute_determinant(matrix):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square")

    det = 0
    for perm in permutations(range(n)):
        # Compute the product for this permutation
        prod = 1.0
        for i in range(n):
            prod = prod * matrix[i][perm[i]]

        # Count the number of inversions to get the sign
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    inversions += 1
        sign = -1 if inversions % 2 else 1

        det += sign * prod

    return det


@pytest.mark.parametrize("d", [2, 3, 4, 5, 6])
def test_compute_determinant_ncpoleon(benchmark, d):
    variables = generate_ncpoleon_variables(d)
    matrix = _build_matrix(variables)
    benchmark(compute_determinant, matrix)


@pytest.mark.benchmark
def test_compute_determinant_for_povm_ncpoleon():
    d = 2
    variables = generate_ncpoleon_variables_for_povm(d)
    matrix = _build_matrix_for_povm(variables)
    compute_determinant(matrix)
