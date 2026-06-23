import logging
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def is_mosek_available():
    if find_spec("mosek") is None:
        return False
    try:
        import mosek

        with mosek.Env() as env:
            env.checkoutlicense(mosek.feature.pts)
        return True
    except mosek.Error:
        logging.warning("MOSEK is installed but no valid license has been found.")
        return False
