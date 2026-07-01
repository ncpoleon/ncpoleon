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

    def __getitem__(self, monomial: PolynomialElements) -> Scalar:
        rewritten_monomial = self._relaxation.rewrite(monomial)
        canonical_monomial, is_adjoint, is_real_valued = self._relaxation.moment_matrices[
            rewritten_monomial.moment_matrix_id
        ].get_canonical(rewritten_monomial)

        if self._primal:
            if is_real_valued:
                return self._model.getVariable(str(canonical_monomial)).level()[0]
            if is_adjoint:
                return (
                    self._model.getVariable(f"{str(canonical_monomial)}_re").level()[0]
                    - self._model.getVariable(f"{str(canonical_monomial)}_im").level()[0] * 1j
                )
            return (
                self._model.getVariable(f"{str(canonical_monomial)}_re").level()[0]
                + self._model.getVariable(f"{str(canonical_monomial)}_im").level()[0] * 1j
            )
        else:
            sign = 1 if self._objective_sense == "min" else -1

            if is_real_valued:
                return self._model.getConstraint(f"M-{canonical_monomial}").dual()[0] * sign
            if is_adjoint:
                return (
                    self._model.getConstraint(f"M-{canonical_monomial}-re").dual()[0]
                    - self._model.getConstraint(f"M-{canonical_monomial}-im").dual()[0] * 1j
                ) * sign
            return (
                self._model.getConstraint(f"M-{canonical_monomial}-re").dual()[0]
                + self._model.getConstraint(f"M-{canonical_monomial}-im").dual()[0] * 1j
            ) * sign

    @property
    def moment_matrix_by_mm_id(
        self,
    ) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]]:
        res = {}

        for id, moment_matrix in self._relaxation.moment_matrices.items():
            size = moment_matrix.size

            if self._primal:
                moment_matrix_level = self._model.getConstraint(f"MM-{id}").level()
            else:
                sign = 1 if self._objective_sense == "max" else -1
                moment_matrix_level = self._model.getVariable(f"Y_{id}").dual() * sign

            if self._relaxation.is_real:
                res[id] = moment_matrix_level.reshape(size, size)
            else:
                moment_matrix_level = moment_matrix_level.reshape(2 * size, 2 * size)
                res[id] = moment_matrix_level[:size, :size] + 1j * moment_matrix_level[size:, :size]

        return res

    @property
    def moment_matrix_multiplier_by_mm_id(
        self,
    ) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]]:
        res = {}

        for id, moment_matrix in self._relaxation.moment_matrices.items():
            size = moment_matrix.size

            if self._primal:
                sign = 1 if self._objective_sense == "min" else -1
                moment_matrix_dual = self._model.getConstraint(f"MM-{id}").dual() * sign
            else:
                moment_matrix_dual = self._model.getVariable(f"Y_{id}").level()

            if self._relaxation.is_real:
                res[id] = moment_matrix_dual.reshape(size, size)
            else:
                moment_matrix_dual = moment_matrix_dual.reshape(2 * size, 2 * size)
                res[id] = moment_matrix_dual[:size, :size] + 1j * moment_matrix_dual[size:, :size]

        return res

    @property
    def localizing_matrices_equality_multipliers_by_mm_id(
        self,
    ) -> dict[
        int,
        list[
            tuple[
                Polynomial[PolynomialElements, Scalar],
                np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]],
            ]
        ],
    ]:
        res = {}

        for (
            id,
            localizing_moment_matrices_equalities_id,
        ) in self._relaxation.localising_moment_matrices_equalities.items():
            to_add = []

            for index, (localizing_moment_matrix, equality_constraint) in enumerate(
                zip(localizing_moment_matrices_equalities_id, self._relaxation.equalities.get(id, []), strict=True)
            ):
                # The equality constraints on symmetric matrices are redundant, and thus Mosek only returns a
                # lower-triangular matrix for the dual, which we have to hermitianize further down
                if self._primal:
                    sign = 1 if self._objective_sense == "min" else -1
                    localizing_moment_matrix_dual = self._model.getConstraint(f"LMME-{id}-{index}").dual() * sign
                else:
                    localizing_moment_matrix_dual = (
                        self._model.getVariable(f"Q_({id}, {index})^0").level()
                        - self._model.getVariable(f"Q_({id}, {index})^1").level()
                    )

                if self._relaxation.is_real:
                    to_hermitianize = localizing_moment_matrix_dual.reshape(
                        localizing_moment_matrix.size, localizing_moment_matrix.size
                    )

                    if self._primal:
                        to_hermitianize = (to_hermitianize + to_hermitianize.T.conj()) / 2

                    to_add.append((equality_constraint, to_hermitianize))
                else:
                    localizing_moment_matrix_dual = localizing_moment_matrix_dual.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_hermitianize = (
                        localizing_moment_matrix_dual[: localizing_moment_matrix.size, : localizing_moment_matrix.size]
                        + 1j
                        * localizing_moment_matrix_dual[
                            localizing_moment_matrix.size :, : localizing_moment_matrix.size
                        ]
                    )

                    if self._primal:
                        to_hermitianize = (to_hermitianize + to_hermitianize.T.conj()) / 2

                    to_add.append((equality_constraint, to_hermitianize))

            res[id] = to_add

        return res

    @property
    def localizing_matrices_inequality_by_mm_id(
        self,
    ) -> dict[
        int,
        list[
            tuple[
                Polynomial[PolynomialElements, Scalar],
                np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]],
            ]
        ],
    ]:
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
                    sign = 1 if self._objective_sense == "max" else -1
                    localizing_moment_matrix_level = self._model.getVariable(f"P_({id}, {index})").dual() * sign

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
    ) -> dict[
        int,
        list[
            tuple[
                Polynomial[PolynomialElements, Scalar],
                np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]],
            ]
        ],
    ]:
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
                    sign = 1 if self._objective_sense == "min" else -1
                    localizing_moment_matrix_dual = self._model.getConstraint(f"LMMI-{id}-{index}").dual() * sign
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
