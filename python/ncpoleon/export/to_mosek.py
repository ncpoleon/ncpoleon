from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from ncpoleon.relaxations import Realness

try:
    # The following import allows to use dunder methods on MOSEK expressions
    import mosek.fusion.pythonic  # noqa: F401
    from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense, PSDVariable

    if TYPE_CHECKING:
        from mosek.fusion import Expression, SparseMatrix

    _mosek_available = True
except ImportError:
    _mosek_available = False

    if TYPE_CHECKING:
        from mosek.fusion import Expr, Expression, Matrix, Model, PSDVariable, SparseMatrix

from ncpoleon._typing import MonomialType, Scalar

if TYPE_CHECKING:
    from ncpoleon.relaxations import BaseSdpRelaxation, MomentMatrix


logger = logging.getLogger(__name__)


class _ComplexExpr:
    """Pair of MOSEK Expr objects representing a complex-valued Expr.

    An `imag` of `None` means an imaginary part that is identically zero, as opposed to an expression that merely
    happens to evaluate to zero. It lets the arithmetic below skip terms that can never contribute, and lets callers
    tell "there is no imaginary part" from "there is one, and it is worth constraining".
    """

    def __init__(self, real: Expression, imag: Expression | None):
        self.real = real
        self.imag = imag

    def __mul__(self, scalar: float | complex) -> _ComplexExpr:
        if isinstance(scalar, complex):
            re, im = scalar.real, scalar.imag
        else:
            re, im = float(scalar), 0.0

        if self.imag is None:
            return _ComplexExpr(Expr.mul(re, self.real), Expr.mul(im, self.real) if im != 0.0 else None)

        return _ComplexExpr(
            Expr.sub(Expr.mul(re, self.real), Expr.mul(im, self.imag)),
            Expr.add(Expr.mul(re, self.imag), Expr.mul(im, self.real)),
        )

    def __add__(self, other: _ComplexExpr) -> _ComplexExpr:
        real = Expr.add(self.real, other.real)

        if self.imag is None:
            return _ComplexExpr(real, other.imag)
        if other.imag is None:
            return _ComplexExpr(real, self.imag)

        return _ComplexExpr(real, Expr.add(self.imag, other.imag))

    def conj(self) -> _ComplexExpr:
        if self.imag is None:
            return self

        return _ComplexExpr(self.real, Expr.mul(-1.0, self.imag))


def convert_row_col_data_to_mosek_symmetric_matrix(
    position_matrix: tuple[list[int], list[int], list[float]], size: int
) -> Matrix:
    rows, cols, data = position_matrix

    return Matrix.sparse(size, size, rows, cols, data)


# TODO: add the docstring
def convert_row_col_data_to_mosek_hermitian_matrix(
    position_matrix: tuple[list[int], list[int], Sequence[complex]], size: int
) -> tuple[Matrix, Matrix, Matrix]:
    rows, cols, data = position_matrix

    return (
        Matrix.sparse(size, size, rows, cols, [x.real for x in data]),
        Matrix.sparse(size, size, rows, cols, [x.imag for x in data]),
        Matrix.sparse(size, size, rows, cols, [-x.imag for x in data]),
    )


# TODO: add the docstring, say that in the real_antihermitianizing=False case, we consider the matrix multiplied by -i
# instead of i, so that we can add each constraint
def convert_row_col_data_to_mosek_hermitianized_matrix(
    position_matrix: tuple[list[int], list[int], Sequence[complex]], size: int, *, real_hermitianizing: bool
) -> SparseMatrix:
    rows, cols, data = position_matrix
    data_re = [x.real for x in data]
    data_im = [x.imag for x in data]

    if real_hermitianizing:
        neg_data_im = [-x.imag for x in data]
        real_part: SparseMatrix = Matrix.sparse(size, size, rows + cols, cols + rows, data_re + data_re)
        imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_im + neg_data_im)
        neg_imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, neg_data_im + data_im)
    else:
        neg_data_re = [-x.real for x in data]
        real_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_im + data_im)
        imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, neg_data_re + data_re)
        neg_imag_part = Matrix.sparse(size, size, rows + cols, cols + rows, data_re + neg_data_re)

    return Matrix.sparse([[real_part, neg_imag_part], [imag_part, real_part]])


def real_moment_matrix_to_mosek(
    moment_matrix: MomentMatrix[MonomialType, float],
    mapped_variables: dict[MonomialType, Expression],
) -> Expression:
    """Build a (localising) moment matrix of a real-valued problem as a symmetric MOSEK expression."""
    # The accumulator starts as `0` and is folded with `Expr.add`, which promotes the seed on the first iteration. It
    # is therefore still an `int` only for a moment matrix with no entry at all, which `get_relaxation` never builds,
    # hence the cast on the return path
    mosek_moment_matrix = 0

    for monomial, (position_matrix, _realness) in moment_matrix.as_row_col_data_format().items():
        matrix = convert_row_col_data_to_mosek_symmetric_matrix(position_matrix, moment_matrix.size)
        mosek_moment_matrix = Expr.add(mosek_moment_matrix, Expr.mul(mapped_variables[monomial], matrix))

    return cast("Expression", mosek_moment_matrix)


def complex_moment_matrix_to_mosek(
    moment_matrix: MomentMatrix[MonomialType, complex],
    mapped_variables: dict[MonomialType, _ComplexExpr],
) -> Expression:
    """Build a (localising) moment matrix of a complex-valued problem as its real symmetric embedding.

    A hermitian matrix `R + iI` is PSD if and only if the real symmetric matrix `[[R, -I], [I, R]]` is, so the real
    and the imaginary parts are accumulated separately and stacked at the end.
    """
    mosek_moment_matrix_re = 0
    mosek_moment_matrix_im = 0

    for monomial, (position_matrix, realness) in moment_matrix.as_row_col_data_format().items():
        matrix_re, matrix_im, _matrix_im_neg = convert_row_col_data_to_mosek_hermitian_matrix(
            position_matrix, moment_matrix.size
        )
        variable = mapped_variables[monomial]

        if realness == Realness.Real:
            # The position matrix holds both orientations already, so the contribution is `x * P` with `x` real
            mosek_moment_matrix_re = Expr.add(mosek_moment_matrix_re, Expr.mul(variable.real, matrix_re))
            mosek_moment_matrix_im = Expr.add(mosek_moment_matrix_im, Expr.mul(variable.real, matrix_im))
        else:
            # The position matrix holds the canonical orientation only: the adjoint monomial sits at the conjugate
            # transpose and carries the conjugate variable, so the contribution is `x * P + conj(x) * P^dagger`
            variable_imag = variable.imag
            assert variable_imag is not None, "only a monomial whose realness is Real has no imaginary part"

            mosek_moment_matrix_re = Expr.add(
                mosek_moment_matrix_re,
                Expr.sub(
                    Expr.add(
                        Expr.mul(variable.real, matrix_re),
                        Expr.mul(variable.real, matrix_re.transpose()),
                    ),
                    Expr.add(
                        Expr.mul(variable_imag, matrix_im),
                        Expr.mul(variable_imag, matrix_im.transpose()),
                    ),
                ),
            )
            mosek_moment_matrix_im = Expr.add(
                mosek_moment_matrix_im,
                Expr.add(
                    Expr.sub(
                        Expr.mul(variable.real, matrix_im),
                        Expr.mul(variable.real, matrix_im.transpose()),
                    ),
                    Expr.sub(
                        Expr.mul(variable_imag, matrix_re),
                        Expr.mul(variable_imag, matrix_re.transpose()),
                    ),
                ),
            )

    # Every entry turned out to be real: the embedding would be block-diagonal with both blocks equal to the real
    # part, which is PSD if and only if that part is, so we can return it directly
    if isinstance(mosek_moment_matrix_im, int):
        return cast("Expression", mosek_moment_matrix_re)

    return Expr.vstack(
        [
            Expr.hstack([mosek_moment_matrix_re, Expr.mul(-1.0, mosek_moment_matrix_im)]),
            Expr.hstack([mosek_moment_matrix_im, mosek_moment_matrix_re]),
        ]
    )


def get_mosek_symmetric_psd_variable(model: Model, name: str, size: int) -> PSDVariable:
    return model.variable(name, Domain.inPSDCone(size))


def get_mosek_hermitian_psd_variable(model: Model, name: str, size: int) -> PSDVariable:
    """A hermitian PSD matrix of the given size, as the real symmetric PSD matrix of twice that size embedding it."""
    return model.variable(name, Domain.inPSDCone(2 * size))


def fill_real_primal_model(
    model: Model,
    sdp: BaseSdpRelaxation[MonomialType, float],
    objective_direction: str,
) -> None:
    """Add the primal form of a real-valued relaxation to `model`.

    Every moment is real, so every monomial maps to a single unbounded MOSEK variable.
    """
    mapped_variables: dict[MonomialType, Expression] = {}

    for moment_matrix_id, moment_matrix in sdp.moment_matrices.items():
        for monomial in moment_matrix.as_row_col_data_format():
            mapped_variables[monomial] = model.variable(str(monomial), Domain.unbounded())

        mosek_moment_matrix = real_moment_matrix_to_mosek(moment_matrix, mapped_variables)
        model.constraint(
            f"MM-{moment_matrix_id}", mosek_moment_matrix, Domain.inPSDCone(mosek_moment_matrix.getShape()[0])
        )
        logger.debug(f"Added moment matrix PSD constraint for moment matrix id {moment_matrix_id}.")

    for moment_matrix_id, equality_moment_matrices in sdp.localising_moment_matrices_equalities.items():
        for equality_index, (_generator, equality_moment_matrix) in enumerate(equality_moment_matrices):
            for poly_index, poly in enumerate(equality_moment_matrix):
                changed = sdp.change_variables(poly, mapped_variables)
                model.constraint(f"ME-{moment_matrix_id}-{equality_index}-{poly_index}", changed, Domain.equalsTo(0))
                logger.debug(f"Added constraint {changed} == 0.")

    for moment_matrix_id, inequality_moment_matrices in sdp.localising_moment_matrices_inequalities.items():
        for inequality_index, inequality_moment_matrix in enumerate(inequality_moment_matrices):
            localising_matrix = real_moment_matrix_to_mosek(inequality_moment_matrix, mapped_variables)
            model.constraint(
                f"LMMI-{moment_matrix_id}-{inequality_index}",
                localising_matrix,
                Domain.inPSDCone(localising_matrix.getShape()[0]),
            )
            logger.debug(f"Added constraint {localising_matrix} >= 0 for moment matrix id {moment_matrix_id}.")

    for index, (poly, value) in enumerate(sdp.moment_equalities):
        changed = sdp.change_variables(poly, mapped_variables)
        model.constraint(f"ME-{index}", changed, Domain.equalsTo(value))
        logger.debug(f"Added constraint {changed} == {value}.")

    for index, (poly, value) in enumerate(sdp.moment_inequalities):
        changed = sdp.change_variables(poly, mapped_variables)
        model.constraint(f"MI-{index}", changed, Domain.greaterThan(value))
        logger.debug(f"Added constraint {changed} >= {value}.")

    model.objective(
        ObjectiveSense.Minimize if objective_direction == "min" else ObjectiveSense.Maximize,
        sdp.change_variables(sdp.objective, mapped_variables),
    )


def fill_complex_primal_model(
    model: Model,
    sdp: BaseSdpRelaxation[MonomialType, complex],
    objective_direction: str,
) -> None:
    """Add the primal form of a complex-valued relaxation to `model`.

    Every monomial maps to a `_ComplexExpr`, whatever its realness: the coefficients are complex, so even the moment
    of a self-adjoint monomial takes part in complex arithmetic. The realness is carried by the imaginary part, which
    is `None` exactly when the moment is real.
    """
    mapped_variables: dict[MonomialType, _ComplexExpr] = {}

    for moment_matrix_id, moment_matrix in sdp.moment_matrices.items():
        for monomial, (_position_matrix, realness) in moment_matrix.as_row_col_data_format().items():
            if realness == Realness.Real:
                mapped_variables[monomial] = _ComplexExpr(model.variable(str(monomial), Domain.unbounded()), None)
            else:
                mapped_variables[monomial] = _ComplexExpr(
                    model.variable(f"{monomial}_re", Domain.unbounded()),
                    model.variable(f"{monomial}_im", Domain.unbounded()),
                )

        mosek_moment_matrix = complex_moment_matrix_to_mosek(moment_matrix, mapped_variables)
        model.constraint(
            f"MM-{moment_matrix_id}", mosek_moment_matrix, Domain.inPSDCone(mosek_moment_matrix.getShape()[0])
        )
        logger.debug(f"Added moment matrix PSD constraint for moment matrix id {moment_matrix_id}.")

    for moment_matrix_id, equality_moment_matrices in sdp.localising_moment_matrices_equalities.items():
        for equality_index, (_generator, equality_moment_matrix) in enumerate(equality_moment_matrices):
            for poly_index, poly in enumerate(equality_moment_matrix):
                changed = sdp.change_variables(poly, mapped_variables)
                model.constraint(
                    f"ME-{moment_matrix_id}-{equality_index}-{poly_index}_re", changed.real, Domain.equalsTo(0.0)
                )
                logger.debug(f"Added constraint {changed.real} == 0.0.")

                if changed.imag is not None:
                    model.constraint(
                        f"ME-{moment_matrix_id}-{equality_index}-{poly_index}_im", changed.imag, Domain.equalsTo(0.0)
                    )
                    logger.debug(f"Added constraint {changed.imag} == 0.0.")

    for moment_matrix_id, inequality_moment_matrices in sdp.localising_moment_matrices_inequalities.items():
        for index, inequality_moment_matrix in enumerate(inequality_moment_matrices):
            localising_matrix = complex_moment_matrix_to_mosek(inequality_moment_matrix, mapped_variables)
            model.constraint(
                f"LMMI-{moment_matrix_id}-{index}",
                localising_matrix,
                Domain.inPSDCone(localising_matrix.getShape()[0]),
            )
            logger.debug(f"Added constraint {localising_matrix} >= 0 for moment matrix id {moment_matrix_id}.")

    for index, (poly, value) in enumerate(sdp.moment_equalities):
        changed = sdp.change_variables(poly, mapped_variables)
        model.constraint(f"ME-{index}_re", changed.real, Domain.equalsTo(value.real))
        logger.debug(f"Added constraint {changed.real} == {value.real}.")

        if changed.imag is not None:
            model.constraint(f"ME-{index}_im", changed.imag, Domain.equalsTo(value.imag))
            logger.debug(f"Added constraint {changed.imag} == {value.imag}.")

    for index, (poly, value) in enumerate(sdp.moment_inequalities):
        # A moment inequality always has a real bound, so it constrains the real part
        changed = sdp.change_variables(poly, mapped_variables)
        model.constraint(f"MI-{index}", changed.real, Domain.greaterThan(value))
        logger.debug(f"Added constraint {changed.real} >= {value}.")

    # The objective is hermitian, so its imaginary part is identically zero
    objective = sdp.change_variables(sdp.objective, mapped_variables)
    model.objective(
        ObjectiveSense.Minimize if objective_direction == "min" else ObjectiveSense.Maximize,
        objective.real,
    )


def fill_real_dual_model(
    model: Model,
    sdp: BaseSdpRelaxation[MonomialType, float],
    objective_direction: str,
) -> None:
    """Add the dual form of a real-valued relaxation to `model`.

    Every moment is real, so every multiplier is a real symmetric matrix and every monomial contributes exactly one
    constraint row.
    """
    operator_inequalities = sdp.localising_moment_matrices_inequalities
    operator_equalities = sdp.localising_moment_matrices_equalities

    moment_inequalities_coefficients = [
        (sdp.get_coefficients_by_canonical(poly)[0], scalar) for (poly, scalar) in sdp.moment_inequalities
    ]
    moment_equalities_coefficients = [
        (sdp.get_coefficients_by_canonical(poly)[0], scalar) for (poly, scalar) in sdp.moment_equalities
    ]
    objective_coefficients, _ = sdp.get_coefficients_by_canonical(sdp.objective)

    lambdas = []
    objective = 0.0

    for m, (_, scalar_inequality) in enumerate(moment_inequalities_coefficients):
        new_variable = model.variable(f"lambda_{m}", Domain.greaterThan(0.0))
        lambdas.append(new_variable)
        objective = Expr.add(objective, Expr.mul(new_variable, scalar_inequality))
        logger.debug(f"Added dual variable lambda_{m} >= 0 for moment inequality number {m}.")

    nus = []

    for n, (_, scalar_equality) in enumerate(moment_equalities_coefficients):
        new_variable = model.variable(f"nu_{n}")
        nus.append(new_variable)
        objective = Expr.add(objective, Expr.mul(new_variable, scalar_equality))
        logger.debug(f"Added dual variable nu_{n} for moment equality number {n}.")

    if objective_direction == "max":
        model.objective(ObjectiveSense.Minimize, -objective)
    else:
        model.objective(ObjectiveSense.Maximize, objective)

    for moment_matrix_index, moment_matrix in sdp.moment_matrices.items():
        Y = get_mosek_symmetric_psd_variable(model, f"Y_{moment_matrix_index}", moment_matrix.size)
        logger.debug(f"Added PSD variable Y_{moment_matrix_index} of size {moment_matrix.size}.")

        Ps = [
            get_mosek_symmetric_psd_variable(
                model, f"P_{(moment_matrix_index, inequality_index)}", inequality_localizing_matrix.size
            )
            for inequality_index, inequality_localizing_matrix in enumerate(operator_inequalities[moment_matrix_index])
        ]
        logger.debug(f"Added {len(Ps)} PSD variable(s) P_* for moment matrix {moment_matrix_index}.")

        Qs = []

        for equality_index, (_generator, equality_as_moments) in enumerate(operator_equalities[moment_matrix_index]):
            for moment_id in range(len(equality_as_moments)):
                Qs.append(model.variable(f"nu_{(moment_matrix_index, equality_index, moment_id)}"))
                logger.debug(
                    f"Added dual variable nu_{(moment_matrix_index, equality_index, moment_id)} for operator equality "
                    f"number {equality_index}."
                )

        for monomial, (position_matrix, _realness) in moment_matrix.as_row_col_data_format().items():
            F = convert_row_col_data_to_mosek_symmetric_matrix(position_matrix, moment_matrix.size)
            constraint_row = Expr.dot(Y, F)

            for multiplier, localizing_matrix, localizing_matrix_as_row_col in zip(
                Ps,
                operator_inequalities[moment_matrix_index],
                [
                    localizing_matrix.as_row_col_data_format()
                    for localizing_matrix in operator_inequalities[moment_matrix_index]
                ],
            ):
                position_matrix_localizing, localizing_realness = localizing_matrix_as_row_col.get(
                    monomial, (None, Realness.Real)
                )

                if position_matrix_localizing is not None:
                    assert localizing_realness == Realness.Real
                    G = convert_row_col_data_to_mosek_symmetric_matrix(
                        position_matrix_localizing, localizing_matrix.size
                    )
                    constraint_row = Expr.add(constraint_row, Expr.dot(multiplier, G))

            for lambda_m, (coefficients, _scalar) in zip(lambdas, moment_inequalities_coefficients, strict=True):
                beta = coefficients.get(monomial, 0.0)
                constraint_row = Expr.add(constraint_row, Expr.mul(lambda_m, beta))

            operator_equalities_split = [
                (sdp.get_coefficients_by_canonical(poly)[0], 0.0)
                for (_generator, polys) in sdp.localising_moment_matrices_equalities[moment_matrix_index]
                for poly in polys
            ]

            for nu_n, (coefficients, _scalar) in zip(
                nus + Qs, moment_equalities_coefficients + operator_equalities_split, strict=True
            ):
                zeta = coefficients.get(monomial, 0.0)
                constraint_row = Expr.add(constraint_row, Expr.mul(nu_n, zeta))

            mu = objective_coefficients.get(monomial, 0.0)
            model.constraint(
                f"M-{monomial}", constraint_row, Domain.equalsTo(mu if objective_direction == "min" else -mu)
            )
            logger.debug(f"Added dual constraint for monomial {monomial}.")


def hermitian_dot_as_complex_expr(
    multiplier: Expression,
    position_matrix: tuple[list[int], list[int], Sequence[complex]],
    size: int,
    realness: Realness,
) -> _ComplexExpr:
    """Dot a dual multiplier with the embedding of one monomial's position matrix.

    The 1/2 factors compensate the doubling introduced by representing a hermitian matrix as a real symmetric one of
    twice the size, which preserves the dot product.
    """
    if realness == Realness.Real:
        # The position matrix is hermitian but the moment is real, so there is nothing to constrain on the imaginary
        # part
        matrix_re, matrix_im, matrix_im_neg = convert_row_col_data_to_mosek_hermitian_matrix(position_matrix, size)

        return _ComplexExpr(
            Expr.mul(Expr.dot(multiplier, Matrix.sparse([[matrix_re, matrix_im_neg], [matrix_im, matrix_re]])), 1 / 2),
            None,
        )

    # The position matrix holds the canonical orientation only, so it is hermitianized first, once for each part
    return _ComplexExpr(
        Expr.mul(
            Expr.dot(
                multiplier,
                convert_row_col_data_to_mosek_hermitianized_matrix(position_matrix, size, real_hermitianizing=True),
            ),
            1 / 2,
        ),
        Expr.mul(
            Expr.dot(
                multiplier,
                convert_row_col_data_to_mosek_hermitianized_matrix(position_matrix, size, real_hermitianizing=False),
            ),
            1 / 2,
        ),
    )


def fill_complex_dual_model(
    model: Model,
    sdp: BaseSdpRelaxation[MonomialType, complex],
    objective_direction: str,
) -> None:
    """Add the dual form of a complex-valued relaxation to `model`.

    The multipliers are hermitian, so they are embedded as real symmetric matrices of twice their size, and every dot
    product against such an embedding is halved to preserve it. Each monomial accumulates a single complex constraint
    row: a self-adjoint monomial has no imaginary part and yields one real constraint, any other yields two, its
    moment and the moment of its adjoint being independent.
    """
    operator_inequalities = sdp.localising_moment_matrices_inequalities
    operator_equalities = sdp.localising_moment_matrices_equalities

    split_moment_inequalities = [
        (sdp.get_coefficients_by_canonical(poly), scalar) for (poly, scalar) in sdp.moment_inequalities
    ]
    split_moment_equalities = [
        (sdp.get_coefficients_by_canonical(poly), scalar) for (poly, scalar) in sdp.moment_equalities
    ]
    objective_coefficients_real, objective_coefficients_complex = sdp.get_coefficients_by_canonical(sdp.objective)

    lambdas = []
    objective = 0.0

    for m, (_, scalar_inequality) in enumerate(split_moment_inequalities):
        new_variable = model.variable(f"lambda_{m}", Domain.greaterThan(0.0))
        lambdas.append(new_variable)
        objective = Expr.add(objective, Expr.mul(new_variable, scalar_inequality))
        logger.debug(f"Added dual variable lambda_{m} >= 0 for moment inequality number {m}.")

    nus = []

    for n, (_, scalar_equality) in enumerate(split_moment_equalities):
        new_variable = _ComplexExpr(model.variable(f"nu_{n}^re"), model.variable(f"nu_{n}^im"))
        nus.append(new_variable)
        objective = Expr.add(objective, (new_variable.conj() * scalar_equality).real)
        logger.debug(f"Added dual variable nu_{n} for moment equality number {n}.")

    if objective_direction == "max":
        model.objective(ObjectiveSense.Minimize, -objective)
    else:
        model.objective(ObjectiveSense.Maximize, objective)

    for moment_matrix_index, moment_matrix in sdp.moment_matrices.items():
        Y = get_mosek_hermitian_psd_variable(model, f"Y_{moment_matrix_index}", moment_matrix.size)
        logger.debug(f"Added PSD variable Y_{moment_matrix_index} of size {moment_matrix.size}.")

        Ps = [
            get_mosek_hermitian_psd_variable(
                model, f"P_{(moment_matrix_index, inequality_index)}", inequality_localizing_matrix.size
            )
            for inequality_index, inequality_localizing_matrix in enumerate(operator_inequalities[moment_matrix_index])
        ]
        logger.debug(f"Added {len(Ps)} PSD variable(s) P_* for moment matrix {moment_matrix_index}.")

        Qs = []

        for equality_index, (_generator, equality_as_moments) in enumerate(operator_equalities[moment_matrix_index]):
            for poly_index in range(len(equality_as_moments)):
                Qs.append(
                    _ComplexExpr(
                        model.variable(f"nu_{(moment_matrix_index, equality_index, poly_index)}^re"),
                        model.variable(f"nu_{(moment_matrix_index, equality_index, poly_index)}^im"),
                    )
                )
                logger.debug(
                    f"Added dual variable nu_{(moment_matrix_index, equality_index)} for operator equality number "
                    f"{equality_index}."
                )

        for monomial, (position_matrix, realness) in moment_matrix.as_row_col_data_format().items():
            constraint_row = hermitian_dot_as_complex_expr(Y, position_matrix, moment_matrix.size, realness)

            for multiplier, localizing_matrix, localizing_matrix_as_row_col in zip(
                Ps,
                operator_inequalities[moment_matrix_index],
                [
                    localizing_matrix.as_row_col_data_format()
                    for localizing_matrix in operator_inequalities[moment_matrix_index]
                ],
            ):
                position_matrix_localizing, localizing_realness = localizing_matrix_as_row_col.get(
                    monomial, (None, Realness.Real)
                )

                if position_matrix_localizing is not None:
                    assert localizing_realness == realness
                    constraint_row = constraint_row + hermitian_dot_as_complex_expr(
                        multiplier, position_matrix_localizing, localizing_matrix.size, realness
                    )

            for lambda_m, ((real_coefficients, complex_coefficients), _scalar) in zip(
                lambdas, split_moment_inequalities, strict=True
            ):
                if realness == Realness.Real:
                    beta = complex(real_coefficients.get(monomial, 0.0)).real
                    constraint_row = constraint_row + _ComplexExpr(Expr.mul(lambda_m, beta), None)
                else:
                    # A moment inequality is hermitian, so the adjoint monomial carries the conjugate coefficient and
                    # the pair contributes twice the real part of `beta * y`, on the scale of the objective row below
                    beta_complex, _beta_conj = complex_coefficients.get(monomial, (0.0 + 0.0j, 0.0 + 0.0j))
                    constraint_row = constraint_row + _ComplexExpr(
                        Expr.mul(Expr.mul(lambda_m, beta_complex.real), 2.0),
                        Expr.mul(Expr.mul(lambda_m, beta_complex.imag), 2.0),
                    )

            operator_equalities_split = [
                (sdp.get_coefficients_by_canonical(poly), 0.0)
                for (_generator, polys) in sdp.localising_moment_matrices_equalities[moment_matrix_index]
                for poly in polys
            ]

            for nu_n, ((real_coefficients, complex_coefficients), _scalar) in zip(
                nus + Qs, split_moment_equalities + operator_equalities_split, strict=True
            ):
                if realness == Realness.Real:
                    zeta = real_coefficients.get(monomial, 0.0 + 0.0j)
                    constraint_row = constraint_row + _ComplexExpr((nu_n.conj() * zeta).real, None)
                else:
                    delta, eps = complex_coefficients.get(monomial, (0.0 + 0.0j, 0.0 + 0.0j))
                    nu_n_imag = nu_n.imag
                    assert nu_n_imag is not None

                    constraint_row = constraint_row + _ComplexExpr(
                        Expr.add(
                            Expr.mul(nu_n.real, (delta + eps).real),
                            Expr.mul(nu_n_imag, (delta + eps).imag),
                        ),
                        Expr.sub(
                            Expr.mul(nu_n.real, (delta - eps).imag),
                            Expr.mul(nu_n_imag, (delta - eps).real),
                        ),
                    )

            sign = 1.0 if objective_direction == "min" else -1.0

            if realness == Realness.Real:
                # The objective is hermitian, so a self-adjoint monomial always carries a real coefficient
                mu = complex(objective_coefficients_real.get(monomial, 0.0)).real
                model.constraint(f"M-{monomial}", constraint_row.real, Domain.equalsTo(sign * mu))
                logger.debug(f"Added dual constraint for monomial {monomial}.")
            else:
                alpha, _alpha_conj = objective_coefficients_complex.get(monomial, (0.0 + 0.0j, 0.0 + 0.0j))
                constraint_row_imag = constraint_row.imag
                assert constraint_row_imag is not None

                # The factor 2 compensates the doubling of the real symmetric embedding
                model.constraint(f"M-{monomial}-re", constraint_row.real, Domain.equalsTo(sign * 2 * alpha.real))
                model.constraint(f"M-{monomial}-im", constraint_row_imag, Domain.equalsTo(sign * 2 * alpha.imag))
                logger.debug(f"Added dual constraints for monomial {monomial}.")


# FIXME: this can probably be simplified by defining ComplexVariables and HermitianVariables just like PICOS
#  More generally, we could probably provide a blanket implementation for the export, given that the user
#  provides the function with what's a real variable, a complex one, a symmetric one, a hermitian one, and such
#  that the variables can be multiplied together, be taken the trace of, etc.
def to_mosek(
    sdp: BaseSdpRelaxation[MonomialType, Scalar],
    objective_direction: str,
    *,
    primal: bool,
    verbosity: Literal[0] | Literal[1] | Literal[2] | Literal[3] = 0,
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

    if verbosity > 0:
        M.setSolverParam("log", verbosity)
        M.setLogHandler(sys.stdout)

    for param, value in model_kwargs.items():
        M.setSolverParam(param, value)

    # `sdp.is_real` is what decides whether `Scalar` is a float or a complex, which no type checker can follow, hence
    # the casts below. They are the only ones needed: each function called here is monomorphic in the scalar type
    if primal:
        logger.info("Exporting to a primal MOSEK problem.")

        if sdp.is_real:
            fill_real_primal_model(M, cast("BaseSdpRelaxation[MonomialType, float]", sdp), objective_direction)
        else:
            fill_complex_primal_model(M, cast("BaseSdpRelaxation[MonomialType, complex]", sdp), objective_direction)
    else:
        logger.info("Exporting to a dual MOSEK problem.")

        if sdp.is_real:
            fill_real_dual_model(M, cast("BaseSdpRelaxation[MonomialType, float]", sdp), objective_direction)
        else:
            fill_complex_dual_model(M, cast("BaseSdpRelaxation[MonomialType, complex]", sdp), objective_direction)

    logger.info("MOSEK problem created.")

    return M
