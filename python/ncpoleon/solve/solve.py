import logging

from ncpoleon.export import to_mosek, to_picos
from ncpoleon.solve import MosekSolution
from ncpoleon.solve.solution import BaseSolution

from .utils import automatic_solver_detection

logger = logging.getLogger(__name__)


def solve(
    relaxation,
    objective_direction: str,
    *,
    force_primal: bool = False,
    solver: str = "auto",
    **solver_parameters,
) -> BaseSolution:
    if solver == "auto":
        solver = automatic_solver_detection()

    if solver not in ["mosek", "picos"]:
        raise ValueError(f"{solver} isn't an acceptable value for the solver. Possible values are mosek and picos.")

    if solver == "mosek":
        model = to_mosek(relaxation, objective_direction, primal=force_primal, **solver_parameters)
        model.solve()

        return MosekSolution(relaxation, model, primal=force_primal)
    elif solver == "picos":
        if "picos_solver" in solver_parameters:  # TODO: to put in the documentation of this function
            picos_solver = solver_parameters.pop("picos_solver")
            solver_parameters["solver"] = picos_solver

        problem = to_picos(relaxation, objective_direction, primal=force_primal, **solver_parameters)
        problem.solve()

        return PicosSolution(relaxation, problem, primal=force_primal)
    else:
        raise ValueError(f"{solver} isn't a valid solver. Possible solver are mosek and picos.")
