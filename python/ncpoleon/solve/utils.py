import logging
from importlib.util import find_spec
from typing import cast, overload

import numpy as np
import numpy.typing as npt

from ncpoleon.utils import is_mosek_available

logger = logging.getLogger(__name__)


def automatic_solver_detection() -> str:
    if is_mosek_available():
        return "mosek"

    if find_spec("picos") is None:
        raise ImportError("No solver has been found. Tried: mosek, picos.")

    return "picos"


# FIXME: change to np.ndarray directly in the type hints, so tht we can specify the shape
@overload
def sos_vectors_of_hermitian_matrix(
    matrix: npt.NDArray[np.float64], cutoff: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
@overload
def sos_vectors_of_hermitian_matrix(
    matrix: npt.NDArray[np.complex128], cutoff: float
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]: ...


def sos_vectors_of_hermitian_matrix(
    matrix: npt.NDArray[np.float64 | np.complex128], cutoff: float
) -> tuple[npt.NDArray[np.float64 | np.complex128], npt.NDArray[np.float64 | np.complex128]]:
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
    result = (positive_eigvals * positive_eigvecs).T.conj(), (negative_eigvals * negative_eigvecs).T.conj()

    return cast(tuple[npt.NDArray[np.float64 | np.complex128], npt.NDArray[np.float64 | np.complex128]], result)
