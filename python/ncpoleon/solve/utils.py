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
    # `-1j * matrix` is hermitian, so `matrix = sum_k 1j * mu_k * w_k @ w_k^†`
    eigvals, eigvecs = np.linalg.eigh(-1j * matrix)

    cutoff_mask = np.abs(eigvals) >= cutoff
    eigvecs = eigvecs[:, cutoff_mask]
    eigvals = eigvals[cutoff_mask]

    if np.isrealobj(matrix):
        # A real antisymmetric matrix has its eigenvalues in pairs ±mu with conjugate eigenvectors, so the positive half
        # alone holds the whole matrix, as 2 * mu * (Re w @ Im w^T - Im w @ Re w^T)
        mask = eigvals > 0
        scale = np.sqrt(2 * eigvals[mask])
        positive_eigvecs = eigvecs[:, mask]
        result = scale * positive_eigvecs.imag, scale * positive_eigvecs.real
    else:
        # Each eigenvector stands on its own: the pair (s * w, 1j * sign(mu) * s * w) with s = sqrt(|mu| / 2) gives back
        # 1j * mu * w @ w^†
        scale = np.sqrt(np.abs(eigvals) / 2)
        result = scale * eigvecs, 1j * np.sign(eigvals) * scale * eigvecs

    return cast(tuple[RealOrComplexMatrix, RealOrComplexMatrix], result)
