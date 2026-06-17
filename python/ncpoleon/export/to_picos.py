from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from scipy.sparse import coo_matrix

try:
    import picos as pc

    _picos_available = True
except ImportError:
    _picos_available = False
    if TYPE_CHECKING:
        import picos as pc

from ncpoleon._typing import PolynomialElements, Scalar

if TYPE_CHECKING:
    from ncpoleon.relaxations import BaseSdpRelaxation

logger = logging.getLogger(__name__)


@overload
def convert_row_col_data_to_coo_matrix(
    position_matrix: tuple[list[int], list[int], list[Scalar]], size: int
) -> coo_matrix: ...


@overload
def convert_row_col_data_to_coo_matrix(position_matrix: None, size: int) -> None: ...


def convert_row_col_data_to_coo_matrix(
    position_matrix: tuple[list[int], list[int], list[Scalar]] | None, size: int
) -> coo_matrix | None:
    if position_matrix is None:
        return None

    rows, cols, data = position_matrix

    return coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(size, size))


def to_picos(
    sdp: BaseSdpRelaxation[PolynomialElements, Scalar],
    objective_direction: str,
    *,
    primal: bool,
    **problem_kwargs: Any,
) -> pc.modeling.Problem:
    r"""Export a relaxation to PICOS.

    :param sdp: The relaxation to be converted to PICOS, generated with :func:`~ncpoleon.relaxations.get_relaxation`.
    :param objective_direction: Could be either "min" or "max", defines the optimization sense. Note that this is the
        optimization direction of the **primal** problem.
    :param primal: If `True`, then the problem is exported in its primal form. Otherwise, it is exported in its dual
        form.
    :param \**problem_kwargs: Any additional keyword arguments to be passed to :class:`picos.Problem` at instantiation.
    :return: A :class:`picos.Problem` object corresponding to the problem the user has specified.
    """
    if not _picos_available:
        raise ImportError("picos is required for to_picos but is not installed. Install it with: pip install picos")
    if objective_direction not in ["min", "max"]:
        raise ValueError(
            f"The only supported objective directions are min and max, but {objective_direction} was provided."
        )

    problem = pc.Problem(**problem_kwargs)

    if primal:
        logger.info("Exporting to a primal PICOS problem.")
        mapped_variables = {}

        for moment_matrix_id, moment_matrix in sdp.moment_matrices.items():
            mapped_moment_matrix = {}
            for monomial, (position_matrix, position_matrix_conj) in moment_matrix.as_row_col_data_format().items():
                new_variable = (
                    pc.RealVariable(str(monomial))
                    if position_matrix_conj is None
                    else pc.ComplexVariable(str(monomial))
                )

                mapped_moment_matrix[new_variable] = (
                    convert_row_col_data_to_coo_matrix(position_matrix, moment_matrix.size),
                    convert_row_col_data_to_coo_matrix(position_matrix_conj, moment_matrix.size),
                )

                mapped_variables[monomial] = new_variable

            G = pc.sum(
                mon * pos_matrix + (0 if pos_matrix_conj is None else mon.conj() * pos_matrix_conj)
                for mon, (pos_matrix, pos_matrix_conj) in mapped_moment_matrix.items()
            )
            problem.add_constraint(G >> 0)
            logger.debug(f"Added moment matrix PSD constraint for moment matrix id {moment_matrix_id}.")

        for moment_matrix_id, equality_moment_matrices in sdp.localising_moment_matrices_equalities.items():
            for equality_moment_matrix in equality_moment_matrices:
                new_localising_matrix = pc.sum(
                    mapped_variables[mon] * convert_row_col_data_to_coo_matrix(pos_matrix, equality_moment_matrix.size)
                    + (
                        0
                        if pos_matrix_conj is None
                        else mapped_variables[mon].conj()
                        * convert_row_col_data_to_coo_matrix(pos_matrix_conj, equality_moment_matrix.size)
                    )
                    for mon, (pos_matrix, pos_matrix_conj) in equality_moment_matrix.as_row_col_data_format().items()
                )
                problem.add_constraint(new_localising_matrix == 0)
                logger.debug(f"Added constraint {new_localising_matrix} == 0 for moment matrix id {moment_matrix_id}.")

        for moment_matrix_id, inequality_moment_matrices in sdp.localising_moment_matrices_inequalities.items():
            for inequality_moment_matrix in inequality_moment_matrices:
                new_localising_matrix = pc.sum(
                    mapped_variables[mon]
                    * convert_row_col_data_to_coo_matrix(pos_matrix, inequality_moment_matrix.size)
                    + (
                        0
                        if pos_matrix_conj is None
                        else mapped_variables[mon].conj()
                        * convert_row_col_data_to_coo_matrix(pos_matrix_conj, inequality_moment_matrix.size)
                    )
                    for mon, (pos_matrix, pos_matrix_conj) in inequality_moment_matrix.as_row_col_data_format().items()
                )
                problem.add_constraint(new_localising_matrix >> 0)
                logger.debug(f"Added constraint {new_localising_matrix} ≽ 0 for moment matrix id {moment_matrix_id}.")

        # FIXME: We should instead pass the mapped variables to the relaxation, which could then return all the moment
        #  at once. That would reduce conversion costs and potentially allow to add logging for individual constraints
        problem.add_list_of_constraints(
            sdp.change_variables(poly, mapped_variables) == value for poly, value in sdp.moment_equalities
        )
        problem.add_list_of_constraints(
            sdp.change_variables(poly, mapped_variables) >= value for poly, value in sdp.moment_inequalities
        )

        problem.set_objective(objective_direction, sdp.change_variables(sdp.objective, mapped_variables))
    else:
        logger.info("Exporting to a dual MOSEK problem.")

        is_problem_real_valued = sdp.is_real
        operator_inequalities = sdp.localising_moment_matrices_inequalities
        operator_equalities = sdp.localising_moment_matrices_equalities
        split_objective_re, split_objective_im = sdp.split_into_real_and_imaginary_parts(sdp.objective)
        assert split_objective_im is None

        split_moment_inequalities = [
            (sdp.split_into_real_and_imaginary_parts(poly), scalar) for (poly, scalar) in sdp.moment_inequalities
        ]
        split_moment_equalities = [
            (sdp.split_into_real_and_imaginary_parts(poly), scalar) for (poly, scalar) in sdp.moment_equalities
        ]

        lambdas = []
        objective = 0.0

        for m, (_, scalar_inequality) in enumerate(split_moment_inequalities):
            new_variable = pc.RealVariable(f"lambda_{m}", lower=0.0)
            lambdas.append(new_variable)
            objective += new_variable * scalar_inequality
            logger.debug(f"Added dual variable lambda_{m} >= 0 for moment inequality number {m}.")

        nus = []

        for n, (_, scalar_equality) in enumerate(split_moment_equalities):
            if is_problem_real_valued:
                new_variable = pc.RealVariable(f"nu_{n}")
                nus.append(new_variable)
                objective += new_variable * scalar_equality
            else:
                new_variable = pc.ComplexVariable(f"nu_{n}")
                nus.append(new_variable)
                objective += (new_variable.conj * scalar_equality).real
            logger.debug(f"Added dual variable nu_{n} for moment equality number {n}.")

        if objective_direction == "max":
            problem.set_objective("min", -objective)
        else:
            problem.set_objective("max", objective)

        variable_builder = pc.SymmetricVariable if is_problem_real_valued else pc.HermitianVariable

        for moment_matrix_index, moment_matrix in sdp.moment_matrices.items():
            Y = variable_builder(f"Y_{moment_matrix_index}", (moment_matrix.size, moment_matrix.size))
            problem.add_constraint(Y >> 0)
            logger.debug(f"Added PSD variable Y_{moment_matrix_index} of size {moment_matrix.size}.")

            Ps = [
                variable_builder(
                    f"P_{(moment_matrix_index, inequality_index)}",
                    (inequality_localizing_matrix.size, inequality_localizing_matrix.size),
                )
                for inequality_index, inequality_localizing_matrix in enumerate(
                    operator_inequalities[moment_matrix_index]
                )
            ]
            problem.add_list_of_constraints(P >> 0 for P in Ps)
            logger.debug(f"Added {len(Ps)} PSD variable(s) P_* for moment matrix {moment_matrix_index}.")

            Qs = [
                variable_builder(
                    f"Q_{(moment_matrix_index, equality_index)}",
                    (equality_localizing_matrix.size, equality_localizing_matrix.size),
                )
                for equality_index, equality_localizing_matrix in enumerate(operator_equalities[moment_matrix_index])
            ]

            logger.debug(f"Added {len(Qs)} free Hermitian variable Q_* for moment matrix {moment_matrix_index}.")

            # Precompute localizing matrix row-col formats outside the monomial loop.
            localizing_row_cols = [
                [
                    localizing_matrix.as_row_col_data_format()
                    for localizing_matrix in operator_inequalities[moment_matrix_index]
                ],
                [
                    localizing_matrix.as_row_col_data_format()
                    for localizing_matrix in operator_equalities[moment_matrix_index]
                ],
            ]

            for monomial, (pos_matrix, pos_matrix_conj) in moment_matrix.as_row_col_data_format().items():
                F = convert_row_col_data_to_coo_matrix(pos_matrix, moment_matrix.size)
                new_constraint = pc.trace(Y * F)

                for lagrange_mutlipliers, localizing_matrices, precomputed_row_cols in zip(
                    [Ps, Qs],
                    [operator_inequalities[moment_matrix_index], operator_equalities[moment_matrix_index]],
                    localizing_row_cols,
                ):
                    for multiplier, localizing_matrix, localizing_matrix_as_row_col in zip(
                        lagrange_mutlipliers, localizing_matrices, precomputed_row_cols
                    ):
                        pos_matrix_localizing, _pos_matrix_localizing_conj = localizing_matrix_as_row_col.get(
                            monomial, (None, None)
                        )

                        if pos_matrix_localizing is not None:
                            G = convert_row_col_data_to_coo_matrix(pos_matrix_localizing, localizing_matrix.size)
                            new_constraint += pc.trace(multiplier * G)

                for lambda_m, ((poly_re, poly_im), _) in zip(lambdas, split_moment_inequalities):
                    assert poly_im is None
                    beta_re, minus_beta_im = poly_re.get(monomial, (None, None))

                    # beta_re can only be None if the monomial isn't present in the moment inequality constraint
                    if beta_re is not None:
                        new_constraint += lambda_m * (beta_re - minus_beta_im * 1j)

                for nu_n, ((poly_re, poly_im), _) in zip(nus, split_moment_equalities):
                    if pos_matrix_conj is None:
                        if is_problem_real_valued:
                            assert poly_im is None

                        delta_re, delta_im = poly_re.get(monomial, (None, None))

                        if delta_re is not None:
                            assert delta_im is None
                            new_constraint += nu_n.real * delta_re
                    else:
                        delta_plus_eps_re, minus_delta_minus_eps_im = poly_re.get(monomial, (0.0, None))
                        minus_delta_minus_eps_im = 0.0 if minus_delta_minus_eps_im is None else minus_delta_minus_eps_im

                        if poly_im is not None:
                            delta_plus_eps_im, delta_minus_eps_re = poly_im.get(monomial, (0.0, None))
                        else:
                            delta_minus_eps_re = 0.0 if delta_minus_eps_re is None else delta_minus_eps_re

                        delta_plus_eps = delta_plus_eps_re + delta_plus_eps_im * 1j
                        delta_minus_eps = delta_minus_eps_re - minus_delta_minus_eps_im * 1j
                        delta = (delta_plus_eps + delta_minus_eps) / 2
                        eps = (delta_plus_eps - delta_minus_eps) / 2

                        new_constraint += nu_n.conj * delta + nu_n * eps.conjugate()

                alpha_re, alpha_im = split_objective_re.get(monomial, (0.0, None))

                if alpha_im is None:
                    alpha_im = 0.0

                alpha = alpha_re + alpha_im * 1j

                if objective_direction == "min":
                    problem.add_constraint(new_constraint == alpha)
                else:
                    problem.add_constraint(new_constraint == -alpha)

                logger.debug(f"Added dual constraint for monomial {monomial}.")

    logger.info("PICOS problem created.")
    return problem
