from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ncpoleon._typing import MonomialType, Scalar

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
    coefficients: tuple[dict[MonomialType, Scalar], dict[MonomialType, tuple[complex, complex]]],
) -> bool:
    """Whether a polynomial constrains nothing once its monomials become moment variables.

    This covers the zero polynomial, but also, in a real-valued relaxation, an anti-hermitian one such
    as ``x2 x1 - x1 x2``: a monomial and its adjoint share a single moment variable there, so every
    canonical coefficient cancels and the equality reduces to ``0 == 0``. Such a moment gets neither a
    constraint nor a multiplier -- PICOS drops a variable that appears in no constraint row, and MOSEK
    would otherwise carry a free variable whose value is arbitrary.

    It takes the polynomial already split by ``BaseSdpRelaxation.get_coefficients_by_canonical`` rather than the
    polynomial itself, because that split crosses into Rust and the callers that keep a vacuous moment out of the
    dual go on to need the very same coefficients.
    """
    real_coefficients, complex_coefficients = coefficients

    return not any(real_coefficients.values()) and not any(
        canonical or adjoint for canonical, adjoint in complex_coefficients.values()
    )
