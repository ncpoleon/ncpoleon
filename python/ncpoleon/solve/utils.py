import logging
from importlib.util import find_spec
from typing import cast, overload

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def automatic_solver_detection() -> str:
    if find_spec("mosek") is not None:
        try:
            import mosek

            with mosek.Env() as env:
                env.checkoutlicense(mosek.feature.pts)
            return "mosek"
        except mosek.Error:
            logging.warning("MOSEK is installed but no valid license has been found, skipping.")

    if find_spec("picos") is None:
        raise ImportError("No solver has been found. Tried: mosek, picos.")

    return "picos"


@overload
def sqrtm_sdp_matrix(matrix: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...
@overload
def sqrtm_sdp_matrix(matrix: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]: ...


def sqrtm_sdp_matrix(
    matrix: npt.NDArray[np.floating | np.complexfloating],
) -> npt.NDArray[np.floating | np.complexfloating]:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.sqrt(np.clip(eigvals, 0, None))
    result = (eigvecs * eigvals) @ eigvecs.conj().T

    return cast(npt.NDArray[np.floating | np.complexfloating], result)
