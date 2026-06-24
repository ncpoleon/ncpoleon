import logging

from ncpoleon.export import to_mosek, to_picos
from ncpoleon.solve import MosekSolution, PicosSolution
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

    if solver == "mosek":
        model = to_mosek(relaxation, objective_direction, primal=force_primal, **solver_parameters)
        model.solve()

        return MosekSolution(relaxation, model, primal=force_primal, objective_sense=objective_direction)
    elif solver == "picos":
        problem, constraints = to_picos(
            relaxation, objective_direction, primal=force_primal, **solver_parameters
        )
        problem.solve()

        return PicosSolution(relaxation, problem, constraints, primal=force_primal)
    elif solver.startswith("picos-"):  # TODO: to put in the docstring of this function
        problem, constraints = to_picos(
            relaxation, objective_direction, primal=force_primal, solver=solver[6:], **solver_parameters
        )
        problem.solve()

        return PicosSolution(relaxation, problem, constraints, primal=force_primal)
    else:
        raise ValueError(f"{solver} isn't a valid solver. Possible solvers are mosek, picos and picos-{{solver}}.")
