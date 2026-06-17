import logging

from .utils import automatic_solver_detection

logger = logging.getLogger(__name__)


def solve(relaxation, solver=None, solver_parameters=None):
    if solver is None:
        solver = automatic_solver_detection()

    if solver not in ["mosek", "picos"]:
        raise ValueError(f"{solver} isn't an acceptable value for the solver. Possible values are mosek and picos.")


