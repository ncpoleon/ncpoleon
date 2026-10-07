from __future__ import annotations

from importlib.util import find_spec
from typing import overload, cast

import numpy as np

from ncpoleon._typing import RealOrComplexMatrix
from ncpoleon.utils import is_mosek_available


def automatic_solver_detection() -> str:
    if is_mosek_available():
        return "mosek"

    if find_spec("picos") is None:
        raise ImportError("No solver has been found. Tried: mosek, picos.")

    return "picos"


@overload
def sos_vectors_of_hermitian_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]], cutoff: float
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]]]: ...
@overload
def sos_vectors_of_hermitian_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex128]], cutoff: float
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]], np.ndarray[tuple[int, int], np.dtype[np.complex128]]
]: ...


def sos_vectors_of_hermitian_matrix(
    matrix: RealOrComplexMatrix, cutoff: float
) -> tuple[RealOrComplexMatrix, RealOrComplexMatrix]:
    eigvals, eigvecs = np.linalg.eigh(matrix)

    # Remove small eigvals
    cutoff_mask = np.abs(eigvals) >= cutoff
    eigvecs = eigvecs[:, cutoff_mask]
    eigvals = eigvals[cutoff_mask]

    # Split positive and negative eigvals
    mask = eigvals >= 0
    positive_eigvecs = eigvecs[:, mask]
    positive_eigvals = np.sqrt(eigvals[mask])
    negative_eigvecs = eigvecs[:, ~mask]
    negative_eigvals = np.sqrt(-eigvals[~mask])
    result = (positive_eigvals * positive_eigvecs), (negative_eigvals * negative_eigvecs)

    return cast(tuple[RealOrComplexMatrix, RealOrComplexMatrix], result)


@overload
def sos_pair_vectors_of_antihermitian_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]], cutoff: float
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float64]], np.ndarray[tuple[int, int], np.dtype[np.float64]]]: ...
@overload
def sos_pair_vectors_of_antihermitian_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex128]], cutoff: float
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.complex128]], np.ndarray[tuple[int, int], np.dtype[np.complex128]]
]: ...


def sos_pair_vectors_of_antihermitian_matrix(
    matrix: RealOrComplexMatrix, cutoff: float
) -> tuple[RealOrComplexMatrix, RealOrComplexMatrix]:
    """Columns ``P`` and ``Q`` such that ``matrix = Q @ P^† - P @ Q^†``, real whenever ``matrix`` is."""
    # `-1j * matrix` is hermitian, so `matrix = sum_k 1j * mu_k * w_k @ w_k^†`, and its SoS vectors are the columns
    # `v_k = sqrt(|mu_k|) * w_k`, split by the sign of `mu_k`
    positive, negative = sos_vectors_of_hermitian_matrix(
        cast("np.ndarray[tuple[int, int], np.dtype[np.complex128]]", -1j * matrix), cutoff
    )

    if np.isrealobj(matrix):
        # A real antisymmetric matrix has its eigenvalues in pairs ±mu with conjugate eigenvectors, so the positive half
        # alone holds the whole matrix, as 2 * mu * (Re w @ Im w^T - Im w @ Re w^T)
        result = np.sqrt(2) * positive.imag, np.sqrt(2) * positive.real
    else:
        # Each vector stands on its own: the pair (v / sqrt(2), ±1j * v / sqrt(2)), with the sign of mu, gives back
        # 1j * mu * w @ w^†
        result = np.hstack([negative, positive]) / np.sqrt(2), 1j * np.hstack([-negative, positive]) / np.sqrt(2)

    return cast(tuple[RealOrComplexMatrix, RealOrComplexMatrix], result)
