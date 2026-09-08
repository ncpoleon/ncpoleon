from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ncpoleon._typing import MonomialType, Scalar
    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation

logger = logging.getLogger(__name__)


def is_mosek_available():
    if find_spec("mosek") is None:
        return False
    try:
        import mosek
    except ImportError:
        return False

    try:
        with mosek.Env() as env:
            env.checkoutlicense(mosek.feature.pts)
        return True
    except mosek.Error:
        logger.warning("MOSEK is installed but no valid license has been found.")
        return False


def is_vacuous_moment(
    sdp: BaseSdpRelaxation[MonomialType, Scalar], polynomial: Polynomial[MonomialType, Scalar]
) -> bool:
    """Whether a polynomial constrains nothing once its monomials become moment variables.

    This covers the zero polynomial, but also, in a real-valued relaxation, an anti-hermitian one such
    as ``x2 x1 - x1 x2``: a monomial and its adjoint share a single moment variable there, so every
    canonical coefficient cancels and the equality reduces to ``0 == 0``. Such a moment gets neither a
    constraint nor a multiplier -- PICOS drops a variable that appears in no constraint row, and MOSEK
    would otherwise carry a free variable whose value is arbitrary.
    """
    real_coefficients, complex_coefficients = sdp.get_coefficients_by_canonical(polynomial)

    return not any(real_coefficients.values()) and not any(
        canonical or adjoint for canonical, adjoint in complex_coefficients.values()
    )
