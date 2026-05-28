from importlib.util import find_spec
import logging

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
