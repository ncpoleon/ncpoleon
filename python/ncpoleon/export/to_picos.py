from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.sparse import coo_matrix

from ncpoleon.relaxations import Realness
from ncpoleon.utils import is_vacuous_moment

try:
    import picos as pc

    _picos_available = True
except ImportError:
    _picos_available = False
    if TYPE_CHECKING:
        import picos as pc

from ncpoleon._typing import MonomialType, Scalar

if TYPE_CHECKING:
    from ncpoleon.relaxations import BaseSdpRelaxation

logger = logging.getLogger(__name__)


def convert_row_col_data_to_coo_matrix(
    position_matrix: tuple[list[int], list[int], list[Scalar]], size: int
) -> coo_matrix:
    rows, cols, data = position_matrix

    return coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(size, size))


def moment_matrix_entry_to_picos(
    variable: pc.RealVariable | pc.ComplexVariable,
    position_matrix: tuple[list[int], list[int], list[Scalar]],
    realness: Realness,
    size: int,
) -> pc.expressions.Expression:
    """Build the contribution of a single monomial to a (localising) moment matrix.

    A real entry stores both orientations of each position, so its position matrix is used as is. A complex entry only
    stores the positions of the canonical monomial: the adjoint monomial sits at the conjugate transposed positions and
    is associated to the conjugate of the same variable.
    """
    matrix = convert_row_col_data_to_coo_matrix(position_matrix, size)

    if realness == Realness.Real:
        return variable * matrix

    return variable * matrix + variable.conj * matrix.conj().transpose()


def to_picos(
    sdp: BaseSdpRelaxation[MonomialType, Scalar],
    objective_direction: str,
    *,
    primal: bool,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
    **problem_kwargs: Any,
) -> tuple[
    pc.modeling.Problem,
    dict[str, pc.constraints.Constraint],
    dict[str, pc.expressions.Expression],
]:
    r"""Export a relaxation to PICOS.

    :param sdp: The relaxation to be converted to PICOS, generated with :func:`~ncpoleon.relaxations.get_relaxation`.
    :param objective_direction: Could be either "min" or "max", defines the optimization sense. Note that this is the
        optimization direction of the **primal** problem.
    :param primal: If `True`, then the problem is exported in its primal form. Otherwise, it is exported in its dual
        form.
    :param \**problem_kwargs: Any additional keyword arguments to be passed to :class:`picos.Problem` at instantiation.
    :return: A :class:`picos.Problem` object corresponding to the problem the user has specified, a dictionary of
        Constraints to get their dual values later on, and a dictionary of the matrix expressions that were constrained
        to be positive semidefinite, keyed like the Constraints, to get their primal values later on. The latter is
        empty when `primal` is `False`, since the dual export constrains variables rather than expressions.
    """
    if not _picos_available:
        raise ImportError("picos is required for to_picos but is not installed. Install it with: pip install picos")
    if objective_direction not in ["min", "max"]:
        raise ValueError(
            f"The only supported objective directions are min and max, but {objective_direction} was provided."
        )

    problem = pc.Problem(**problem_kwargs, verbosity=verbosity)
    constraints = {}
    # Handing these back keeps the primal matrix values reachable without digging them out of the Constraints: PICOS
    # only puts `lhs` on its affine and LMI constraint classes, not on the `Constraint` base that `add_constraint` is
    # typed to return
    psd_matrices = {}

    if primal:
        logger.info("Exporting to a primal PICOS problem.")
        mapped_variables = {}

        for moment_matrix_id, moment_matrix in sdp.moment_matrices.items():
            moment_matrix_terms = []

            for monomial, (position_matrix, realness) in moment_matrix.as_row_col_data_format().items():
                new_variable = (
                    pc.RealVariable(str(monomial)) if realness == Realness.Real else pc.ComplexVariable(str(monomial))
                )

                moment_matrix_terms.append(
                    moment_matrix_entry_to_picos(new_variable, position_matrix, realness, moment_matrix.size)
                )

                mapped_variables[monomial] = new_variable

            G = pc.sum(moment_matrix_terms)
            psd_matrices[f"MM-{moment_matrix_id}"] = G
            constraints[f"MM-{moment_matrix_id}"] = problem.add_constraint(G >> 0)
            logger.debug(f"Added moment matrix PSD constraint for moment matrix id {moment_matrix_id}.")

        # An operator equality is expanded into one moment equality per entry of its localising matrix, so that a
        # non-hermitian generator is not forced through a hermitian multiplier in the dual
        for moment_matrix_id, equality_moment_matrices in sdp.localising_moment_matrices_equalities.items():
            for equality_index, (_generator, equality_as_moments, _hermiticity) in enumerate(equality_moment_matrices):
                for poly_index, poly in enumerate(equality_as_moments):
                    # A vacuous moment would only add 0 == 0; the solution skips the same indices when reading back
                    if is_vacuous_moment(sdp, poly):
                        continue

                    changed = sdp.change_variables(poly, mapped_variables)
                    constraints[f"ME-{moment_matrix_id}-{equality_index}-{poly_index}"] = problem.add_constraint(
                        changed == 0
                    )
                    logger.debug(f"Added constraint {changed} == 0.")

        for moment_matrix_id, inequality_moment_matrices in sdp.localising_moment_matrices_inequalities.items():
            for index, inequality_moment_matrix in enumerate(inequality_moment_matrices):
                new_localising_matrix = pc.sum(
                    [
                        moment_matrix_entry_to_picos(
                            mapped_variables[mon], pos_matrix, realness, inequality_moment_matrix.size
                        )
                        for mon, (pos_matrix, realness) in inequality_moment_matrix.as_row_col_data_format().items()
                    ]
                )
                psd_matrices[f"LMMI-{moment_matrix_id}-{index}"] = new_localising_matrix
                constraints[f"LMMI-{moment_matrix_id}-{index}"] = problem.add_constraint(new_localising_matrix >> 0)
                logger.debug(f"Added constraint {new_localising_matrix} ≽ 0 for moment matrix id {moment_matrix_id}.")

        # FIXME: We should instead pass the mapped variables to the relaxation, which could then return all the moment
        #  at once. That would reduce conversion costs

        for index, (poly, value) in enumerate(sdp.moment_equalities):
            changed = sdp.change_variables(poly, mapped_variables)
            constraints[f"ME-{index}"] = problem.add_constraint(changed == value)
            logger.debug(f"Added moment constraint {poly} == {value}.")

        for index, (poly, value) in enumerate(sdp.moment_inequalities):
            changed = sdp.change_variables(poly, mapped_variables)

            # A moment inequality always has a real bound, so it constrains the real part: a non-hermitian polynomial
            # yields a complex expression, which PICOS refuses to order.
            constraints[f"MI-{index}"] = problem.add_constraint(changed.real >= value)
            logger.debug(f"Added moment constraint {poly} >= {value}.")

        problem.set_objective(objective_direction, sdp.change_variables(sdp.objective, mapped_variables))
    else:
        logger.info("Exporting to a dual PICOS problem.")

        is_problem_real_valued = sdp.is_real
        operator_inequalities = sdp.localising_moment_matrices_inequalities
        operator_equalities = sdp.localising_moment_matrices_equalities

        split_moment_inequalities = [
            (sdp.get_coefficients_by_canonical(poly), scalar) for (poly, scalar) in sdp.moment_inequalities
        ]
        split_moment_equalities = [
            (sdp.get_coefficients_by_canonical(poly), scalar) for (poly, scalar) in sdp.moment_equalities
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

        objective_coefficients_real, objective_coefficients_complex = sdp.get_coefficients_by_canonical(sdp.objective)

        for moment_matrix_index, moment_matrix in sdp.moment_matrices.items():
            Y = variable_builder(f"Y_{moment_matrix_index}", moment_matrix.size)
            constraints[f"Y_{moment_matrix_index}"] = problem.add_constraint(Y >> 0)
            logger.debug(f"Added PSD variable Y_{moment_matrix_index} of size {moment_matrix.size}.")

            Ps = []

            for inequality_index, inequality_localizing_matrix in enumerate(operator_inequalities[moment_matrix_index]):
                Ps.append(
                    variable_builder(
                        f"P_{(moment_matrix_index, inequality_index)}",
                        inequality_localizing_matrix.size,
                    )
                )
                constraints[f"P_{(moment_matrix_index, inequality_index)}"] = problem.add_constraint(Ps[-1] >> 0)
                logger.debug(f"Added PSD variable(s) P_{(moment_matrix_index, inequality_index)}.")

            # An operator equality is exported as one moment equality per entry of its localising matrix, so its
            # multiplier is a scalar per entry rather than a single hermitian matrix
            Qs = []
            split_operator_equalities = []

            for equality_index, (_generator, equality_as_moments, _hermiticity) in enumerate(
                operator_equalities[moment_matrix_index]
            ):
                for poly_index, poly in enumerate(equality_as_moments):
                    # A vacuous moment appears in no constraint row, so PICOS would prune its
                    # variable and `get_variable` would then fail. Skipping it here and in the coefficients
                    # below keeps the two lists aligned, names keeping their index
                    if is_vacuous_moment(sdp, poly):
                        continue

                    name = f"nu_{(moment_matrix_index, equality_index, poly_index)}"
                    Qs.append(pc.RealVariable(name) if is_problem_real_valued else pc.ComplexVariable(name))
                    split_operator_equalities.append((sdp.get_coefficients_by_canonical(poly), 0.0))
                    logger.debug(f"Added dual variable {name} for operator equality number {equality_index}.")

            # Precompute localizing matrix row-col formats outside the monomial loop.
            inequality_row_cols = [
                localizing_matrix.as_row_col_data_format()
                for localizing_matrix in operator_inequalities[moment_matrix_index]
            ]

            for monomial, (pos_matrix, realness) in moment_matrix.as_row_col_data_format().items():
                F = convert_row_col_data_to_coo_matrix(pos_matrix, moment_matrix.size)
                new_constraint = pc.trace(Y * F)

                for multiplier, localizing_matrix, localizing_matrix_as_row_col in zip(
                    Ps, operator_inequalities[moment_matrix_index], inequality_row_cols
                ):
                    # The realness of the localising entry is not needed here: PICOS handles the Hermitian
                    # multipliers natively, so the same expression covers both cases
                    pos_matrix_localizing, _localizing_realness = localizing_matrix_as_row_col.get(
                        monomial, (None, Realness.Real)
                    )

                    if pos_matrix_localizing is not None:
                        G = convert_row_col_data_to_coo_matrix(pos_matrix_localizing, localizing_matrix.size)
                        new_constraint += pc.trace(multiplier * G)

                for lambda_m, (
                    (poly_moment_ineq_real_monomials_coefficients, poly_moment_ineq_complex_monomials_coefficients),
                    _scalar,
                ) in zip(lambdas, split_moment_inequalities):
                    if realness == Realness.Real:
                        beta = poly_moment_ineq_real_monomials_coefficients.get(monomial, 0.0)
                    else:
                        # A moment inequality is hermitian, so the adjoint monomial carries the conjugate
                        # coefficient and the canonical one alone determines the contribution
                        beta, _beta_conj = poly_moment_ineq_complex_monomials_coefficients.get(
                            monomial, (0.0 + 0.0j, 0.0 + 0.0j)
                        )

                    new_constraint += lambda_m * beta

                for nu_n, (
                    (poly_moment_eq_real_monomials_coefficients, poly_moment_eq_complex_monomials_coefficients),
                    _scalar,
                ) in zip(nus + Qs, split_moment_equalities + split_operator_equalities, strict=True):
                    if is_problem_real_valued:
                        zeta = poly_moment_eq_real_monomials_coefficients.get(monomial, 0.0)
                        new_constraint += nu_n * zeta
                    elif realness == Realness.Real:
                        zeta = poly_moment_eq_real_monomials_coefficients.get(monomial, 0.0)
                        new_constraint += (nu_n.conj * zeta).real
                    else:
                        delta, eps = poly_moment_eq_complex_monomials_coefficients.get(
                            monomial, (0.0 + 0.0j, 0.0 + 0.0j)
                        )
                        new_constraint += (nu_n.conj * delta + nu_n * eps.conjugate()) / 2

                if realness == Realness.Real:
                    alpha = complex(objective_coefficients_real.get(monomial, 0.0)).real
                else:
                    alpha, _alpha_conj = objective_coefficients_complex.get(monomial, (0.0 + 0.0j, 0.0 + 0.0j))

                if objective_direction == "min":
                    constraints[f"M-{monomial}"] = problem.add_constraint(new_constraint == alpha)
                else:
                    constraints[f"M-{monomial}"] = problem.add_constraint(new_constraint == -alpha)

                logger.debug(
                    f"Added dual constraint {new_constraint} == {alpha if objective_direction == 'min' else -alpha} "
                    f"for monomial {monomial}."
                )

    logger.info("PICOS problem created.")
    return problem, constraints, psd_matrices
