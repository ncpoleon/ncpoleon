from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

try:
    # The following import allows to use dunder methods on MOSEK expressions
    import mosek.fusion.pythonic  # noqa: F401
    from mosek.fusion import Domain, Expr, ExprMulScalarConst, Matrix, Model, ObjectiveSense, PSDVariable, SparseMatrix

    _mosek_available = True
except ImportError:
    _mosek_available = False

    if TYPE_CHECKING:
        from mosek.fusion import Expr, ExprMulScalarConst, Matrix, Model, PSDVariable, SparseMatrix

from ncpoleon._typing import PolynomialElements, Scalar

if TYPE_CHECKING:
    from ncpoleon.relaxations import BaseSdpRelaxation, MomentMatrix


logger = logging.getLogger(__name__)


class _ComplexExpr:
    """Pair of MOSEK Expr objects representing a complex-valued Expr"""

    def __init__(self, real: Expr, imag: Expr):
        self.real = real
        self.imag = imag

    def __mul__(self, scalar: float | complex) -> _ComplexExpr:
        if isinstance(scalar, complex):
            re, im = scalar.real, scalar.imag
        else:
            re, im = float(scalar), 0.0

        new_real = Expr.sub(Expr.mul(re, self.real), Expr.mul(im, self.imag))
        new_imag = Expr.add(Expr.mul(re, self.imag), Expr.mul(im, self.real))

        return _ComplexExpr(new_real, new_imag)

    def __add__(self, other: _ComplexExpr) -> _ComplexExpr:
        return _ComplexExpr(
            Expr.add(self.real, other.real),
            Expr.add(self.imag, other.imag),
        )

    def conj(self) -> _ComplexExpr:
        return _ComplexExpr(self.real, Expr.mul(-1.0, self.imag))


def convert_row_col_data_to_mosek_symmetric_matrix(
    position_matrix: tuple[list[int], list[int], list[float]] | None, size: int
) -> Matrix | None:
    if position_matrix is None:
        return None

    rows, cols, data = position_matrix

    return Matrix.sparse(size, size, rows, cols, data)


# TODO: add the docstring
def convert_row_col_data_to_mosek_hermitian_matrix(
    position_matrix: tuple[list[int], list[int], list[complex]] | None, size: int
) -> Matrix | None:
    if position_matrix is None:
        return None

    rows, cols, data = position_matrix
    data_re = []
    data_im = []

    for x in data:
        data_re.append(x.real)
        data_im.append(x.imag)

    return (Matrix.sparse(size, size, rows, cols, data_re), Matrix.sparse(size, size, rows, cols, data_im))


def rust_moment_matrix_to_mosek(
    moment_matrix: MomentMatrix[PolynomialElements, Scalar],
    mapped_variables: dict[PolynomialElements, Expr | _ComplexExpr],
    matrix_builder: Callable[[tuple[list[int], list[int], list[complex]] | None, int], Matrix | None],
) -> Matrix:
    mosek_moment_matrix_re = 0
    mosek_moment_matrix_im = 0

    for mon, (pos_matrix, pos_matrix_conj) in moment_matrix.as_row_col_data_format().items():
        pos_matrix = matrix_builder(pos_matrix, moment_matrix.size)
        pos_matrix_conj = matrix_builder(pos_matrix_conj, moment_matrix.size)

        if pos_matrix_conj is None:
            mosek_moment_matrix_re = Expr.add(mosek_moment_matrix_re, Expr.mul(mapped_variables[mon], pos_matrix))
        else:
            mosek_moment_matrix_re = Expr.add(mosek_moment_matrix_re, Expr.mul(mapped_variables[mon].real, pos_matrix))
            mosek_moment_matrix_re = Expr.add(
                mosek_moment_matrix_re, Expr.mul(mapped_variables[mon].real, pos_matrix_conj)
            )
            mosek_moment_matrix_im = Expr.add(mosek_moment_matrix_im, Expr.mul(mapped_variables[mon].imag, pos_matrix))
            mosek_moment_matrix_im = Expr.add(
                mosek_moment_matrix_im, Expr.mul(Expr.mul(-1, mapped_variables[mon].imag), pos_matrix_conj)
            )

    if isinstance(mosek_moment_matrix_im, int):
        return mosek_moment_matrix_re
    return Expr.vstack(
        [
            Expr.hstack([mosek_moment_matrix_re, Expr.mul(-1.0, mosek_moment_matrix_im)]),
            Expr.hstack([mosek_moment_matrix_im, mosek_moment_matrix_re]),
        ]
    )


def get_mosek_psd_variable(model: Model, name: str, size: int, symmetric: bool) -> PSDVariable:
    return model.variable(name, Domain.inPSDCone(size if symmetric else 2 * size))


def mosek_hermitianize(M_re: SparseMatrix, M_im: SparseMatrix) -> ExprMulScalarConst:
    M_re_plus_M_re_T = Expr.add(Expr.constTerm(M_re), Expr.constTerm(M_re.transpose()))
    M_im_minus_M_im_T = Expr.sub(Expr.constTerm(M_im), Expr.constTerm(M_im.transpose()))

    return Expr.mul(
        Expr.vstack(
            [
                Expr.hstack([M_re_plus_M_re_T, Expr.mul(-1.0, M_im_minus_M_im_T)]),
                Expr.hstack([M_im_minus_M_im_T, M_re_plus_M_re_T]),
            ]
        ),
        1 / 2,
    )


def mosek_antihermitianize(M_re: SparseMatrix, M_im: SparseMatrix) -> ExprMulScalarConst:
    M_re_minus_M_re_T = Expr.sub(Expr.constTerm(M_re), Expr.constTerm(M_re.transpose()))
    minus_M_im_plus_M_im_T = Expr.mul(-1.0, Expr.add(Expr.constTerm(M_im), Expr.constTerm(M_im.transpose())))

    return Expr.mul(
        Expr.vstack(
            [
                Expr.hstack([minus_M_im_plus_M_im_T, Expr.mul(-1.0, M_re_minus_M_re_T)]),
                Expr.hstack([M_re_minus_M_re_T, minus_M_im_plus_M_im_T]),
            ]
        ),
        1 / 2,
    )


# FIXME: this can probably be simplified by defining ComplexVariables and HermitianVariables just like PICOS
#  More generally, we can probably provide a blanket implementation for the export, given that the user
#  provides the function with what's a real variable, a complex one, a symmetric one, a hermitian one, and such
#  that the variables can be multiplied together, be taken the trace of, etc.
def to_mosek(
    sdp: BaseSdpRelaxation[PolynomialElements, Scalar],
    objective_direction: str,
    *,
    primal: bool,
    **model_kwargs: Any,
) -> Model:
    r"""Export a relaxation to MOSEK.

    :param sdp: The relaxation to be converted to MOSEK, generated with :func:`~ncpoleon.relaxations.get_relaxation`.
    :param objective_direction: Could be either "min" or "max", defines the optimization sense. Note that this is the
        optimization direction of the **primal** problem.
    :param primal: If `True`, then the problem is exported in its primal form. Otherwise, it is exported in its dual
        form.
    :param \**problem_kwargs: Any additional keyword arguments to be passed to :class:`mosek.Model` at instantiation.
    :return: A :class:`mosek.Model` object corresponding to the problem the user has specified.
    """
    if not _mosek_available:
        raise ImportError(
            "mosek is required for to_mosek but is not installed. Install it with: pip install mosek. Note that a MOSEK"
            " license is required to use MOSEK."
        )
    if objective_direction not in ["min", "max"]:
        raise ValueError(
            f"The only supported objective directions are min and max, but {objective_direction} was provided."
        )

    M = Model()

    for param, value in model_kwargs.items():
        M.setSolverParam(param, value)

    if primal:
        logger.info("Exporting to a primal MOSEK problem.")

        mapped_variables = {}
        is_problem_real_valued = sdp.is_real
        matrix_builder = (
            convert_row_col_data_to_mosek_symmetric_matrix
            if is_problem_real_valued
            else convert_row_col_data_to_mosek_hermitian_matrix
        )

        for moment_matrix_id, moment_matrix in sdp.moment_matrices.items():
            for monomial, (_position_matrix, position_matrix_conj) in moment_matrix.as_row_col_data_format().items():
                new_variable = (
                    M.variable(str(monomial), Domain.unbounded())
                    if position_matrix_conj is None
                    else _ComplexExpr(
                        M.variable(f"{str(monomial)}_re", Domain.unbounded()),
                        M.variable(f"{str(monomial)}_im", Domain.unbounded()),
                    )
                )

                mapped_variables[monomial] = new_variable

            mosek_moment_matrix = rust_moment_matrix_to_mosek(moment_matrix, mapped_variables, matrix_builder)
            M.constraint(
                f"MM-{moment_matrix_id}", mosek_moment_matrix, Domain.inPSDCone(mosek_moment_matrix.getShape()[0])
            )
            logger.debug(f"Added moment matrix PSD constraint for moment matrix id {moment_matrix_id}.")

        for moment_matrix_id, equality_moment_matrices in sdp.localising_moment_matrices_equalities.items():
            for index, equality_moment_matrix in enumerate(equality_moment_matrices):
                mosek_new_localising_matrix = rust_moment_matrix_to_mosek(
                    equality_moment_matrix, mapped_variables, matrix_builder
                )
                M.constraint(f"LMME-{moment_matrix_id}-{index}", mosek_new_localising_matrix, Domain.equalsTo(0))
                logger.debug(
                    f"Added constraint {mosek_new_localising_matrix} == 0 for moment matrix id {moment_matrix_id}."
                )

        for moment_matrix_id, inequality_moment_matrices in sdp.localising_moment_matrices_inequalities.items():
            for index, inequality_moment_matrix in enumerate(inequality_moment_matrices):
                mosek_new_localising_matrix = rust_moment_matrix_to_mosek(
                    inequality_moment_matrix, mapped_variables, matrix_builder
                )
                M.constraint(
                    f"LMMI-{moment_matrix_id}-{index}",
                    mosek_new_localising_matrix,
                    Domain.inPSDCone(mosek_new_localising_matrix.getShape()[0]),
                )
                logger.debug(
                    f"Added constraint {mosek_new_localising_matrix} ≽ 0 for moment matrix id {moment_matrix_id}."
                )

        for index, (poly, value) in enumerate(sdp.moment_equalities):
            changed = sdp.change_variables(poly, mapped_variables)

            if isinstance(changed, _ComplexExpr):
                M.constraint(f"ME-{index}_re", changed.real, Domain.equalsTo(value.real))
                logger.debug(f"Added constraint {changed.real} == {value.real}.")
                M.constraint(f"ME-{index}_im", changed.imag, Domain.equalsTo(value.imag))
                logger.debug(f"Added constraint {changed.imag} == {value.imag}.")
            else:
                M.constraint(f"ME-{index}", changed, Domain.equalsTo(value))
                logger.debug(f"Added constraint {changed} == {value}.")

        for index, (poly, value) in enumerate(sdp.moment_inequalities):
            changed = sdp.change_variables(poly, mapped_variables)

            if isinstance(changed, _ComplexExpr):
                M.constraint(f"MI-{index}", changed.real, Domain.greaterThan(value))
                logger.debug(f"Added constraint {changed.real} >= {value}.")
            else:
                M.constraint(f"MI-{index}", changed, Domain.greaterThan(value))
                logger.debug(f"Added constraint {changed} >= {value}.")

        objective = sdp.change_variables(sdp.objective, mapped_variables)
        M.objective(
            ObjectiveSense.Minimize if objective_direction == "min" else ObjectiveSense.Maximize,
            objective.real if isinstance(objective, _ComplexExpr) else objective,
        )
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
            new_variable = M.variable(f"lambda_{m}", Domain.greaterThan(0.0))
            lambdas.append(new_variable)
            objective = Expr.add(objective, Expr.mul(new_variable, scalar_inequality))
            logger.debug(f"Added dual variable lambda_{m} >= 0 for moment inequality number {m}.")

        nus = []

        for n, (_, scalar_equality) in enumerate(split_moment_equalities):
            if is_problem_real_valued:
                new_variable = M.variable(f"nu_{n}")
                nus.append(new_variable)
                objective = Expr.add(objective, Expr.mul(new_variable, scalar_equality))
            else:
                new_variable = _ComplexExpr(M.variable(f"nu_{n}^re"), M.variable(f"nu_{n}^im"))
                nus.append(new_variable)
                objective = Expr.add(objective, (new_variable.conj() * scalar_equality).real)
            logger.debug(f"Added dual variable nu_{n} for moment equality number {n}.")

        if objective_direction == "max":
            M.objective(ObjectiveSense.Minimize, -objective)
        else:
            M.objective(ObjectiveSense.Maximize, objective)

        for moment_matrix_index, moment_matrix in sdp.moment_matrices.items():
            Y = get_mosek_psd_variable(M, f"Y_{moment_matrix_index}", moment_matrix.size, is_problem_real_valued)
            logger.debug(f"Added PSD variable Y_{moment_matrix_index} of size {moment_matrix.size}.")

            Ps = [
                get_mosek_psd_variable(
                    M,
                    f"P_{(moment_matrix_index, inequality_index)}",
                    inequality_localizing_matrix.size,
                    is_problem_real_valued,
                )
                for inequality_index, inequality_localizing_matrix in enumerate(
                    operator_inequalities[moment_matrix_index]
                )
            ]
            logger.debug(f"Added {len(Ps)} PSD variable(s) P_* for moment matrix {moment_matrix_index}.")

            Qs = [
                Expr.sub(
                    get_mosek_psd_variable(
                        M,
                        f"Q_{(moment_matrix_index, equality_index)}^0",
                        equality_localizing_matrix.size,
                        is_problem_real_valued,
                    ),
                    get_mosek_psd_variable(
                        M,
                        f"Q_{(moment_matrix_index, equality_index)}^1",
                        equality_localizing_matrix.size,
                        is_problem_real_valued,
                    ),
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
                if is_problem_real_valued:  #  position matrix is symmetric
                    F = convert_row_col_data_to_mosek_symmetric_matrix(pos_matrix, moment_matrix.size)
                    new_constraint = Expr.dot(Y, F)
                elif pos_matrix_conj is None:  # position matrix is symmetric but represented as Hermitian
                    F = convert_row_col_data_to_mosek_symmetric_matrix(pos_matrix, moment_matrix.size)
                    new_constraint = Expr.mul(Expr.dot(Y, Matrix.diag([F, F])), 1 / 2)
                else:  # position matrix only contains the position of the canonical monomial
                    F_re, F_im = convert_row_col_data_to_mosek_hermitian_matrix(pos_matrix, moment_matrix.size)
                    # Even though the trace of the representation is twice the trace of the original matrix, since we
                    # need to consider F + F^dagger in the trace and since the hermitianize function actually returns
                    # the representation of (F + F^dagger) / 2, we can simply consider .dot here without adding the 1/2
                    # factor
                    new_constraint_re = Expr.dot(Y, mosek_hermitianize(F_re, F_im))
                    # The above comment also applies to antihermitianize
                    new_constraint_im = Expr.dot(Y, mosek_antihermitianize(F_re, F_im))

                for lagrange_mutlipliers, localizing_matrices, precomputed_row_cols in zip(
                    [Ps, Qs],
                    [operator_inequalities[moment_matrix_index], operator_equalities[moment_matrix_index]],
                    localizing_row_cols,
                ):
                    for multiplier, localizing_matrix, localizing_matrix_as_row_col in zip(
                        lagrange_mutlipliers, localizing_matrices, precomputed_row_cols
                    ):
                        pos_matrix_localizing, pos_matrix_localizing_conj = localizing_matrix_as_row_col.get(
                            monomial, (None, None)
                        )

                        if pos_matrix_localizing is not None:
                            if is_problem_real_valued:
                                assert pos_matrix_localizing_conj is None
                                G = convert_row_col_data_to_mosek_symmetric_matrix(
                                    pos_matrix_localizing, localizing_matrix.size
                                )
                                new_constraint = Expr.add(new_constraint, Expr.dot(multiplier, G))
                            elif pos_matrix_localizing_conj is None:
                                G = convert_row_col_data_to_mosek_symmetric_matrix(
                                    pos_matrix_localizing, localizing_matrix.size
                                )
                                new_constraint = Expr.add(
                                    new_constraint, Expr.mul(Expr.dot(multiplier, Matrix.diag([G, G])), 1 / 2)
                                )
                            else:
                                G_re, G_im = convert_row_col_data_to_mosek_hermitian_matrix(
                                    pos_matrix_localizing, localizing_matrix.size
                                )
                                new_constraint_re = Expr.add(
                                    new_constraint_re, Expr.dot(multiplier, mosek_hermitianize(G_re, G_im))
                                )
                                new_constraint_im = Expr.add(
                                    new_constraint_im, Expr.dot(multiplier, mosek_antihermitianize(G_re, G_im))
                                )

                for lambda_m, ((poly_re, poly_im), _) in zip(lambdas, split_moment_inequalities):
                    assert poly_im is None
                    beta_re, minus_beta_im = poly_re.get(monomial, (None, None))

                    # beta_re can only be None if the monomial isn't present in the moment inequality constraint
                    if beta_re is not None:
                        if is_problem_real_valued or pos_matrix_conj is None:
                            assert minus_beta_im is None
                            new_constraint = Expr.add(new_constraint, Expr.mul(lambda_m, beta_re))
                        else:
                            assert minus_beta_im is not None
                            new_constraint_re = Expr.add(new_constraint_re, Expr.mul(Expr.mul(lambda_m, beta_re), 2.0))
                            new_constraint_im = Expr.add(
                                new_constraint_im, Expr.mul(Expr.mul(lambda_m, minus_beta_im), 2.0)
                            )

                for nu_n, ((poly_re, poly_im), _) in zip(nus, split_moment_equalities):
                    if pos_matrix_conj is None:
                        if is_problem_real_valued:
                            assert poly_im is None

                        delta_re, delta_im = poly_re.get(monomial, (None, None))

                        if delta_re is not None:
                            assert delta_im is None
                            new_constraint = Expr.add(new_constraint, Expr.mul(nu_n, delta_re))
                    else:
                        delta_plus_eps_re, minus_delta_minus_eps_im = poly_re.get(monomial, (None, None))

                        if poly_im is not None:
                            delta_plus_eps_im, delta_minus_eps_re = poly_im.get(monomial, (None, None))
                        else:
                            delta_plus_eps_im, delta_minus_eps_re = None, None

                        if delta_plus_eps_re is not None:
                            new_constraint_re = Expr.add(new_constraint_re, Expr.mul(nu_n.real, delta_plus_eps_re))

                        if delta_plus_eps_im is not None:
                            new_constraint_re = Expr.add(new_constraint_re, Expr.mul(nu_n.imag, delta_plus_eps_im))

                        if minus_delta_minus_eps_im is not None:
                            new_constraint_im = Expr.add(
                                new_constraint_im, Expr.mul(nu_n.real, minus_delta_minus_eps_im)
                            )

                        if delta_minus_eps_re is not None:
                            new_constraint_im = Expr.add(new_constraint_im, Expr.mul(nu_n.imag, delta_minus_eps_re))

                alpha_re, alpha_im = split_objective_re.get(monomial, (0.0, None))

                if pos_matrix_conj is None:
                    if is_problem_real_valued:
                        assert alpha_im is None
                    if objective_direction == "min":
                        M.constraint(f"M-{monomial}", new_constraint, Domain.equalsTo(alpha_re))
                    else:
                        M.constraint(f"M-{monomial}", new_constraint, Domain.equalsTo(-alpha_re))

                    logger.debug(f"Added dual constraint for monomial {monomial}.")
                else:
                    alpha_im = 0.0 if alpha_im is None else alpha_im

                    if objective_direction == "min":
                        M.constraint(f"M-{monomial}-re", new_constraint_re, Domain.equalsTo(2 * alpha_re))
                        M.constraint(f"M-{monomial}-im", new_constraint_im, Domain.equalsTo(2 * alpha_im))
                    else:
                        M.constraint(f"M-{monomial}-re", new_constraint_re, Domain.equalsTo(-2 * alpha_re))
                        M.constraint(f"M-{monomial}-im", new_constraint_im, Domain.equalsTo(-2 * alpha_im))

                    logger.debug(f"Added dual constraints for monomial {monomial}.")

    logger.info("MOSEK problem created.")
    return M
