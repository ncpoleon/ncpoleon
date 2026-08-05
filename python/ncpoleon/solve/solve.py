import logging
from typing import Literal

from ncpoleon.export import to_mosek, to_picos
from ncpoleon.logging import set_verbosity_level
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
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    **solver_parameters,
) -> BaseSolution:

    set_verbosity_level(verbosity)

    if solver == "auto":
        solver = automatic_solver_detection()

    if solver == "mosek":
        model = to_mosek(relaxation, objective_direction, primal=force_primal, verbosity=verbosity, **solver_parameters)
        model.solve()

        return MosekSolution(relaxation, model, primal=force_primal, objective_sense=objective_direction)
    elif solver == "picos":
        problem, constraints = to_picos(
            relaxation, objective_direction, primal=force_primal, verbosity=verbosity, **solver_parameters
        )
        problem.solve()

        return PicosSolution(relaxation, problem, constraints, primal=force_primal)
    elif solver.startswith("picos-"):  # TODO: to put in the docstring of this function
        problem, constraints = to_picos(
            relaxation,
            objective_direction,
            primal=force_primal,
            verbosity=verbosity,
            solver=solver[6:],
            **solver_parameters,
        )
        problem.solve()

        return PicosSolution(relaxation, problem, constraints, primal=force_primal)
    else:
        raise ValueError(f"{solver} isn't a valid solver. Possible solvers are mosek, picos and picos-{{solver}}.")
