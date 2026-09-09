from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ncpoleon._typing import ComplexMatrix, MonomialType, RealMatrix, RealOrComplexMatrix, Scalar
from ncpoleon.relaxations import Canonicality, Hermiticity, Realness
from ncpoleon.solve.solution import BaseSolution
from ncpoleon.utils import is_vacuous_moment

if TYPE_CHECKING:
    from mosek.fusion import Constraint, Model, Variable

    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation


def _hermitian_from_embedding(embedded: RealMatrix, size: int) -> ComplexMatrix:
    """Recover `R + iI` from the real symmetric embedding `[[R, -I], [I, R]]` that MOSEK optimises over.

    Only a matrix built as that embedding carries its structure exactly, which is the case for the value of
    an expression or variable we constructed. A cone multiplier is any member of the dual cone and carries
    it only up to a member of the orthogonal complement, so it has to be projected onto the structured
    subspace rather than read off two of its blocks. The projection is the identity on a structured matrix.
    """
    real = (embedded[:size, :size] + embedded[size:, size:]) / 2
    imag = (embedded[size:, :size] - embedded[:size, size:]) / 2

    return cast("ComplexMatrix", real + 1j * imag)


class MosekSolution(BaseSolution[MonomialType, Scalar]):
    def __init__(
        self,
        relaxation: BaseSdpRelaxation[MonomialType, Scalar],
        model: Model,
        primal: bool,
        objective_sense: str,
    ):
        self._relaxation = relaxation
        self._model = model
        self._primal = primal

        if objective_sense not in ["min", "max"]:
            raise ValueError(f'objective_sense should be "min" or "max" but {objective_sense} was given.')

        self._objective_sense = objective_sense

    # MOSEK returns None rather than raising when a model holds nothing under the
    # given name, so every lookup of an item the solver is known to have built
    # goes through these two helpers instead of repeating the None check
    def _constraint(self, name: str) -> Constraint:
        constraint = self._model.getConstraint(name)

        if constraint is None:
            raise KeyError(f"The model has no constraint named {name}.")

        return constraint

    def _variable(self, name: str) -> Variable:
        variable = self._model.getVariable(name)

        if variable is None:
            raise KeyError(f"The model has no variable named {name}.")

        return variable

    @property
    def value(self) -> np.float64:
        return self._model.primalObjValue()

    @property
    def relaxation(self) -> BaseSdpRelaxation[MonomialType, Scalar]:
        return self._relaxation

    def __getitem__(self, monomial: MonomialType) -> Scalar:
        rewritten_monomial = self._relaxation.rewrite(monomial)
        canonical_monomial, canonicality, realness = self._moment_matrices[
            rewritten_monomial.moment_matrix_id
        ].get_canonical(rewritten_monomial)

        if self._primal:
            if realness == Realness.Real:
                return self._variable(str(canonical_monomial)).level().item(0)
            if canonicality == Canonicality.Adjoint:
                return cast(
                    Scalar,
                    self._variable(f"{str(canonical_monomial)}_re").level().item(0)
                    - self._variable(f"{str(canonical_monomial)}_im").level().item(0) * 1j,
                )
            return cast(
                Scalar,
                self._variable(f"{str(canonical_monomial)}_re").level().item(0)
                + self._variable(f"{str(canonical_monomial)}_im").level().item(0) * 1j,
            )
        else:
            sign = 1 if self._objective_sense == "min" else -1

            if realness == Realness.Real:
                return self._constraint(f"M-{canonical_monomial}").dual().item(0) * sign
            if canonicality == Canonicality.Adjoint:
                return (
                    cast(
                        Scalar,
                        self._constraint(f"M-{canonical_monomial}-re").dual().item(0)
                        + self._constraint(f"M-{canonical_monomial}-im").dual().item(0) * 1j,
                    )
                    * sign
                )
            return (
                cast(
                    Scalar,
                    self._constraint(f"M-{canonical_monomial}-re").dual().item(0)
                    - self._constraint(f"M-{canonical_monomial}-im").dual().item(0) * 1j,
                )
                * sign
            )

    @property
    def moment_matrix_by_mm_id(
        self,
    ) -> dict[int, RealOrComplexMatrix]:
        res: dict[int, RealOrComplexMatrix] = {}

        for id, moment_matrix in self._moment_matrices.items():
            size = moment_matrix.size

            if self._primal:
                moment_matrix_level = self._constraint(f"MM-{id}").level()
            else:
                sign = 1 if self._objective_sense == "max" else -1
                moment_matrix_level = self._variable(f"Y_{id}").dual() * sign

            if self._relaxation.is_real:
                res[id] = moment_matrix_level.reshape(size, size)
            else:
                moment_matrix_level = moment_matrix_level.reshape(2 * size, 2 * size)

                if not self._primal:  # Needed because of the Hermitian into Symmetric embedding
                    moment_matrix_level *= 2

                res[id] = _hermitian_from_embedding(moment_matrix_level, size)

        return res

    @property
    def moment_matrix_multiplier_by_mm_id(
        self,
    ) -> dict[int, RealOrComplexMatrix]:
        res: dict[int, RealOrComplexMatrix] = {}

        for id, moment_matrix in self._moment_matrices.items():
            size = moment_matrix.size

            if self._primal:
                sign = 1 if self._objective_sense == "min" else -1
                moment_matrix_dual = self._constraint(f"MM-{id}").dual() * sign
            else:
                moment_matrix_dual = self._variable(f"Y_{id}").level()

            if self._relaxation.is_real:
                res[id] = moment_matrix_dual.reshape(size, size)
            else:
                moment_matrix_dual = moment_matrix_dual.reshape(2 * size, 2 * size)

                if self._primal:  # Needed because of the Hermitian into Symmetric embedding
                    moment_matrix_dual *= 2

                res[id] = _hermitian_from_embedding(moment_matrix_dual, size)

        return res

    def _localizing_matrices_equality_multipliers_by_mm_id(
        self, hermiticity: Hermiticity
    ) -> dict[
        int,
        list[
            tuple[
                Polynomial[MonomialType, Scalar],
                list[tuple[Polynomial[MonomialType, Scalar], Scalar]],
                list[MonomialType],
            ]
        ],
    ]:
        res = {}
        # Each of these getters rebuilds its whole map on access, cloning every polynomial and monomial, so they are
        # read once here rather than once per moment matrix
        equalities = self._relaxation.equalities

        for moment_matrix_id, equalities_as_moments in self._relaxation.localising_moment_matrices_equalities.items():
            list_of_equalities: list[
                tuple[
                    Polynomial[MonomialType, Scalar],
                    list[tuple[Polynomial[MonomialType, Scalar], Scalar]],
                    list[MonomialType],
                ]
            ] = []

            for equality_index, (
                (equality_as_polynomial, equality_as_moments, equality_hermiticity),
                (_generator, generating_set),
            ) in enumerate(zip(equalities_as_moments, equalities.get(moment_matrix_id, []), strict=True)):
                if equality_hermiticity != hermiticity:
                    continue

                list_of_moments: list[tuple[Polynomial[MonomialType, Scalar], Scalar]] = []

                for moment_index, moment in enumerate(equality_as_moments):
                    # A vacuous moment got neither a constraint nor a variable, and 0 satisfies it
                    if is_vacuous_moment(self._relaxation.get_coefficients_by_canonical(moment)):
                        list_of_moments.append((moment, cast("Scalar", 0.0)))
                        continue

                    if self._primal:
                        sign = 1 if self._objective_sense == "min" else -1
                        if self._relaxation.is_real:
                            list_of_moments.append(
                                (
                                    moment,
                                    self._constraint(f"ME-{moment_matrix_id}-{equality_index}-{moment_index}").dual()[0]
                                    * sign,
                                )
                            )
                        else:
                            # A moment equality with a real coefficient has no imaginary
                            # part to constrain, so the model may hold the real one alone
                            im_constraint = self._model.getConstraint(
                                f"ME-{moment_matrix_id}-{equality_index}-{moment_index}_im"
                            )

                            if im_constraint is not None:
                                list_of_moments.append(
                                    (
                                        moment,
                                        (
                                            self._constraint(
                                                f"ME-{moment_matrix_id}-{equality_index}-{moment_index}_re"
                                            ).dual()[0]
                                            + im_constraint.dual()[0] * 1j
                                        )
                                        * sign,
                                    )
                                )
                            else:
                                list_of_moments.append(
                                    (
                                        moment,
                                        self._constraint(
                                            f"ME-{moment_matrix_id}-{equality_index}-{moment_index}_re"
                                        ).dual()[0]
                                        * sign,
                                    )
                                )
                    else:
                        if self._relaxation.is_real:
                            list_of_moments.append(
                                (
                                    moment,
                                    self._variable(f"nu_{(moment_matrix_id, equality_index, moment_index)}").level()[0],
                                )
                            )
                        else:
                            list_of_moments.append(
                                (
                                    moment,
                                    self._variable(f"nu_{(moment_matrix_id, equality_index, moment_index)}^re").level()[
                                        0
                                    ]
                                    + self._variable(
                                        f"nu_{(moment_matrix_id, equality_index, moment_index)}^im"
                                    ).level()[0]
                                    * 1j,
                                )
                            )

                list_of_equalities.append((equality_as_polynomial, list_of_moments, generating_set))

            res[moment_matrix_id] = list_of_equalities

        return res

    @property
    def localizing_matrices_inequality_by_mm_id(
        self,
    ) -> dict[
        int,
        list[
            tuple[
                Polynomial[MonomialType, Scalar],
                RealOrComplexMatrix,
                list[MonomialType],
            ]
        ],
    ]:
        res = {}
        # `inequalities` rebuilds its whole map on access, cloning every polynomial and monomial, so it is
        # read once here rather than once per moment matrix
        inequalities = self._relaxation.inequalities

        for (
            id,
            localizing_moment_matrices_inequalities_id,
        ) in self._relaxation.localising_moment_matrices_inequalities.items():
            to_add: list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]] = []

            for index, (localizing_moment_matrix, (inequality_constraint, generating_set)) in enumerate(
                zip(localizing_moment_matrices_inequalities_id, inequalities.get(id, []), strict=True)
            ):
                if self._primal:
                    localizing_moment_matrix_level = self._constraint(f"LMMI-{id}-{index}").level()
                else:
                    sign = 1 if self._objective_sense == "max" else -1
                    localizing_moment_matrix_level = self._variable(f"P_({id}, {index})").dual() * sign

                if self._relaxation.is_real:
                    to_add.append(
                        (
                            inequality_constraint,
                            localizing_moment_matrix_level.reshape(
                                localizing_moment_matrix.size, localizing_moment_matrix.size
                            ),
                            generating_set,
                        )
                    )
                else:
                    localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )

                    if not self._primal:  # Needed because of the Hermitian into Symmetric embedding
                        localizing_moment_matrix_level *= 2

                    to_add.append(
                        (
                            inequality_constraint,
                            _hermitian_from_embedding(localizing_moment_matrix_level, localizing_moment_matrix.size),
                            generating_set,
                        )
                    )

            res[id] = to_add

        return res

    @property
    def localizing_matrices_inequality_multipliers_by_mm_id(
        self,
    ) -> dict[
        int,
        list[
            tuple[
                Polynomial[MonomialType, Scalar],
                RealOrComplexMatrix,
                list[MonomialType],
            ]
        ],
    ]:
        res = {}
        # `inequalities` rebuilds its whole map on access, cloning every polynomial and monomial, so it is
        # read once here rather than once per moment matrix
        inequalities = self._relaxation.inequalities

        for (
            id,
            localizing_moment_matrices_inequalities_id,
        ) in self._relaxation.localising_moment_matrices_inequalities.items():
            to_add: list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]] = []

            for index, (localizing_moment_matrix, (inequality_constraint, generating_set)) in enumerate(
                zip(localizing_moment_matrices_inequalities_id, inequalities.get(id, []), strict=True)
            ):
                if self._primal:
                    sign = 1 if self._objective_sense == "min" else -1
                    localizing_moment_matrix_dual = self._constraint(f"LMMI-{id}-{index}").dual() * sign
                else:
                    localizing_moment_matrix_dual = self._variable(f"P_({id}, {index})").level()

                if self._relaxation.is_real:
                    to_add.append(
                        (
                            inequality_constraint,
                            localizing_moment_matrix_dual.reshape(
                                localizing_moment_matrix.size, localizing_moment_matrix.size
                            ),
                            generating_set,
                        )
                    )
                else:
                    localizing_moment_matrix_dual = localizing_moment_matrix_dual.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )

                    if self._primal:  # Needed because of the Hermitian into Symmetric embedding
                        localizing_moment_matrix_dual *= 2

                    to_add.append(
                        (
                            inequality_constraint,
                            _hermitian_from_embedding(localizing_moment_matrix_dual, localizing_moment_matrix.size),
                            generating_set,
                        )
                    )

            res[id] = to_add

        return res

    @property
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[MonomialType, Scalar], np.float64 | np.complex128]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_equalities):
            if self._primal:
                sign = 1 if self._objective_sense == "min" else -1
                if self._relaxation.is_real:
                    res.append((polynomial_constraint, self._constraint(f"ME-{index}").dual()[0] * sign))
                else:
                    # A moment equality with a real coefficient has no imaginary
                    # part to constrain, so the model may hold the real one alone
                    im_constraint = self._model.getConstraint(f"ME-{index}_im")

                    if im_constraint is not None:
                        res.append(
                            (
                                polynomial_constraint,
                                (self._constraint(f"ME-{index}_re").dual()[0] + im_constraint.dual()[0] * 1j) * sign,
                            )
                        )
                    else:
                        res.append(
                            (
                                polynomial_constraint,
                                self._constraint(f"ME-{index}_re").dual()[0] * sign,
                            )
                        )
            else:
                if self._relaxation.is_real:
                    res.append((polynomial_constraint, self._variable(f"nu_{index}").level()[0]))
                else:
                    res.append(
                        (
                            polynomial_constraint,
                            self._variable(f"nu_{index}^re").level()[0]
                            + self._variable(f"nu_{index}^im").level()[0] * 1j,
                        )
                    )

        return res

    @property
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[MonomialType, Scalar], np.float64]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_inequalities):
            if self._primal:
                sign = 1 if self._objective_sense == "min" else -1
                res.append((polynomial_constraint, self._constraint(f"MI-{index}").dual()[0] * sign))
            else:
                res.append((polynomial_constraint, self._variable(f"lambda_{index}").level()[0]))

        return res
