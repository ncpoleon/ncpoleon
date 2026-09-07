from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from ncpoleon._typing import MonomialType, RealOrComplexMatrix, Scalar
from ncpoleon.utils import is_mosek_available

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ncpoleon.polynomials import Polynomial


def automatic_solver_detection() -> str:
    if is_mosek_available():
        return "mosek"

    if find_spec("picos") is None:
        raise ImportError("No solver has been found. Tried: mosek, picos.")

    return "picos"


@overload
def sos_vectors_of_hermitian_psd_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]], cutoff: float
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]: ...
@overload
def sos_vectors_of_hermitian_psd_matrix(
    matrix: np.ndarray[tuple[int, int], np.dtype[np.complex128]], cutoff: float
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]: ...


def sos_vectors_of_hermitian_psd_matrix(matrix: RealOrComplexMatrix, cutoff: float) -> RealOrComplexMatrix:
    eigvals, eigvecs = np.linalg.eigh(matrix)

    # Remove small eigvals
    cutoff_mask = np.abs(eigvals) >= cutoff
    eigvecs = eigvecs[:, cutoff_mask]
    eigvals = eigvals[cutoff_mask]

    return cast(RealOrComplexMatrix, np.sqrt(eigvals) * eigvecs)
