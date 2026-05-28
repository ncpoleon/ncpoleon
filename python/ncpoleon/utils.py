from importlib.util import find_spec


def is_mosek_available():
    if find_spec("mosek") is None:
        return False
    try:
        import mosek

        with mosek.Env() as env:
            env.checkoutlicense(mosek.feature.pts)
        return True
    except mosek.Error:
        return False
