from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ncpoleon._typing import PolynomialElements, Scalar
from ncpoleon.solve.solution import BaseSolution

if TYPE_CHECKING:
    from mosek.fusion import Model

    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation


class MosekSolution(BaseSolution[PolynomialElements, Scalar]):
    def __init__(
        self,
        relaxation: BaseSdpRelaxation[PolynomialElements, Scalar],
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

    @property
    def value(self) -> np.float64:
        return self._model.primalObjValue()

    @property
    def relaxation(self) -> BaseSdpRelaxation[PolynomialElements, Scalar]:
        return self._relaxation

    def __getitem__(self, monomial: PolynomialElements) -> np.float64 | np.complex128:
        rewritten_monomial = self._relaxation.reduce_monomial(monomial)
        canonical_monomial, is_adjoint, is_real_valued = self._relaxation.moment_matrices[
            rewritten_monomial.moment_matrix_id
        ].get_canonical(rewritten_monomial)

        if self._primal:
            if is_real_valued:
                return self._model.getVariable(str(canonical_monomial)).level()[0]
            if is_adjoint:
                return (
                    self._model.getVariable(f"{str(monomial)}_re").level()[0]
                    - self._model.getVariable(f"{str(monomial)}_im").level()[0] * 1j
                )
            return (
                self._model.getVariable(f"{str(monomial)}_re").level()[0]
                + self._model.getVariable(f"{str(monomial)}_im").level()[0] * 1j
            )
        else:
            sign = 1 if self._objective_sense == "max" else -1

            if is_real_valued:
                return self._model.getConstraint(f"M-{canonical_monomial}").dual()[0] * sign
            if is_adjoint:
                return (
                    self._model.getConstraint(f"M-{monomial}-re").dual()[0]
                    - self._model.getConstraint(f"M-{monomial}-im").dual()[0] * 1j
                ) * sign
            return (
                self._model.getConstraint(f"M-{monomial}-re").dual()[0]
                + self._model.getConstraint(f"M-{monomial}-im").dual()[0] * 1j
            ) * sign

    @property
    def moment_matrix_by_mm_id(self) -> dict[int, np.ndarray]:
        res = {}

        for id, moment_matrix in self._relaxation.moment_matrices.items():
            size = moment_matrix.size

            if self._primal:
                moment_matrix_level = self._model.getConstraint("MM-0").level()
            else:
                moment_matrix_level = self._model.getVariable("Y_0").dual()

            if self._relaxation.is_real:
                res[id] = moment_matrix_level.reshape(size, size)
            else:
                moment_matrix_level = moment_matrix_level.reshape(2 * size, 2 * size)
                res[id] = moment_matrix_level[:size, :size] + 1j * moment_matrix_level[size:, :size]

        return res

    @property
    def moment_matrix_multiplier_by_mm_id(self) -> dict[int, np.ndarray]:
        res = {}

        for id, moment_matrix in self._relaxation.moment_matrices.items():
            size = moment_matrix.size

            if self._primal:
                moment_matrix_dual = self._model.getConstraint("MM-0").dual()
            else:
                moment_matrix_dual = self._model.getVariable("Y_0").level()

            if self._relaxation.is_real:
                res[id] = moment_matrix_dual.reshape(size, size)
            else:
                moment_matrix_dual = moment_matrix_dual.reshape(2 * size, 2 * size)
                res[id] = moment_matrix_dual[:size, :size] + 1j * moment_matrix_dual[size:, :size]

        return res

    @property
    def localizing_matrices_equality_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]:
        res = {}

        for (
            id,
            localizing_moment_matrices_equalities_id,
        ) in self._relaxation.localising_moment_matrices_equalities.items():
            to_add = []

            for index, (localizing_moment_matrix, equality_constraint) in enumerate(
                zip(localizing_moment_matrices_equalities_id, self._relaxation.equalities.get(id, []), strict=True)
            ):
                if self._primal:
                    localizing_moment_matrix_level = self._model.getConstraint(f"LMME-{id}-{index}").level()
                else:
                    localizing_moment_matrix_level = (
                        self._model.getVariable(f"Q_({id}, {index})^0").dual()
                        - self._model.getVariable(f"Q_({id}, {index})^1").dual()
                    )

                if self._relaxation.is_real:
                    to_add.append(
                        (
                            equality_constraint,
                            localizing_moment_matrix_level.reshape(
                                localizing_moment_matrix.size, localizing_moment_matrix.size
                            ),
                        )
                    )
                else:
                    localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_add.append(
                        (
                            equality_constraint,
                            localizing_moment_matrix_level[
                                : localizing_moment_matrix.size, : localizing_moment_matrix.size
                            ]
                            + 1j
                            * localizing_moment_matrix_level[
                                localizing_moment_matrix.size :, : localizing_moment_matrix.size
                            ],
                        )
                    )

            res[id] = to_add

        return res

    @property
    def localizing_matrices_equality_multipliers_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]:
        res = {}

        for (
            id,
            localizing_moment_matrices_equalities_id,
        ) in self._relaxation.localising_moment_matrices_equalities.items():
            to_add = []

            for index, (localizing_moment_matrix, equality_constraint) in enumerate(
                zip(localizing_moment_matrices_equalities_id, self._relaxation.equalities.get(id, []), strict=True)
            ):
                if self._primal:
                    localizing_moment_matrix_dual = self._model.getConstraint(f"LMME-{id}-{index}").dual()
                else:
                    localizing_moment_matrix_dual = (
                        self._model.getVariable(f"Q_({id}, {index})^0").level()
                        - self._model.getVariable(f"Q_({id}, {index})^1").level()
                    )

                if self._relaxation.is_real:
                    to_add.append(
                        (
                            equality_constraint,
                            localizing_moment_matrix_dual.reshape(
                                localizing_moment_matrix.size, localizing_moment_matrix.size
                            ),
                        )
                    )
                else:
                    localizing_moment_matrix_dual = localizing_moment_matrix_dual.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_add.append(
                        (
                            equality_constraint,
                            localizing_moment_matrix_dual[
                                : localizing_moment_matrix.size, : localizing_moment_matrix.size
                            ]
                            + 1j
                            * localizing_moment_matrix_dual[
                                localizing_moment_matrix.size :, : localizing_moment_matrix.size
                            ],
                        )
                    )

            res[id] = to_add

        return res

    @property
    def localizing_matrices_inequality_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]:
        res = {}

        for (
            id,
            localizing_moment_matrices_inequalities_id,
        ) in self._relaxation.localising_moment_matrices_inequalities.items():
            to_add = []

            for index, (localizing_moment_matrix, inequality_constraint) in enumerate(
                zip(localizing_moment_matrices_inequalities_id, self._relaxation.inequalities.get(id, []), strict=True)
            ):
                if self._primal:
                    localizing_moment_matrix_level = self._model.getConstraint(f"LMMI-{id}-{index}").level()
                else:
                    localizing_moment_matrix_level = self._model.getVariable(f"P_({id}, {index})").dual()

                if self._relaxation.is_real:
                    to_add.append(
                        (
                            inequality_constraint,
                            localizing_moment_matrix_level.reshape(
                                localizing_moment_matrix.size, localizing_moment_matrix.size
                            ),
                        )
                    )
                else:
                    localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_add.append(
                        (
                            inequality_constraint,
                            localizing_moment_matrix_level[
                                : localizing_moment_matrix.size, : localizing_moment_matrix.size
                            ]
                            + 1j
                            * localizing_moment_matrix_level[
                                localizing_moment_matrix.size :, : localizing_moment_matrix.size
                            ],
                        )
                    )

            res[id] = to_add

        return res

    @property
    def localizing_matrices_inequality_multipliers_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[PolynomialElements, Scalar], np.ndarray]]]:
        res = {}

        for (
            id,
            localizing_moment_matrices_inequalities_id,
        ) in self._relaxation.localising_moment_matrices_inequalities.items():
            to_add = []

            for index, (localizing_moment_matrix, inequality_constraint) in enumerate(
                zip(localizing_moment_matrices_inequalities_id, self._relaxation.inequalities.get(id, []), strict=True)
            ):
                if self._primal:
                    localizing_moment_matrix_dual = self._model.getConstraint(f"LMMI-{id}-{index}").dual()
                else:
                    localizing_moment_matrix_dual = self._model.getVariable(f"P_({id}, {index})").level()

                if self._relaxation.is_real:
                    to_add.append(
                        (
                            inequality_constraint,
                            localizing_moment_matrix_dual.reshape(
                                localizing_moment_matrix.size, localizing_moment_matrix.size
                            ),
                        )
                    )
                else:
                    localizing_moment_matrix_dual = localizing_moment_matrix_dual.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_add.append(
                        (
                            inequality_constraint,
                            localizing_moment_matrix_dual[
                                : localizing_moment_matrix.size, : localizing_moment_matrix.size
                            ]
                            + 1j
                            * localizing_moment_matrix_dual[
                                localizing_moment_matrix.size :, : localizing_moment_matrix.size
                            ],
                        )
                    )

            res[id] = to_add

        return res

    @property
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.float64 | np.complex128]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_equalities):
            if self._primal:
                if self._relaxation.is_real:
                    res.append((polynomial_constraint, self._model.getConstraint(f"ME-{index}").dual()[0]))
                else:
                    res.append(
                        (
                            polynomial_constraint,
                            self._model.getConstraint(f"ME-{index}_re").dual()[0]
                            + self._model.getConstraint(f"ME-{index}_im").dual()[0] * 1j,
                        )
                    )
            else:
                if self._relaxation.is_real:
                    res.append((polynomial_constraint, self._model.getVariable(f"nu_{index}").level()[0]))
                else:
                    res.append(
                        (
                            polynomial_constraint,
                            self._model.getVariable(f"nu_{index}^re").level()[0]
                            + self._model.getVariable(f"nu_{index}^im").level()[0] * 1j,
                        )
                    )

        return res

    @property
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.float64]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_inequalities):
            if self._primal:
                res.append((polynomial_constraint, self._model.getConstraint(f"MI-{index}").dual()[0]))
            else:
                res.append((polynomial_constraint, self._model.getVariable(f"lambda_{index}").level()[0]))

        return res
