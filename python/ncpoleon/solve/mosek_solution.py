from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ncpoleon.solve.solution import BaseSolution

if TYPE_CHECKING:
    from mosek.fusion import Model

    from ncpoleon.relaxations import (
        ComplexValuedCommutativeSdpRelaxation,
        ComplexValuedNonCommutativeSdpRelaxation,
        RealValuedCommutativeSdpRelaxation,
        RealValuedNonCommutativeSdpRelaxation,
    )


class MosekSolution(BaseSolution):
    def __init__(
        self,
        relaxation: RealValuedCommutativeSdpRelaxation
        | ComplexValuedCommutativeSdpRelaxation
        | RealValuedNonCommutativeSdpRelaxation
        | ComplexValuedNonCommutativeSdpRelaxation,
        model: Model,
        primal: bool,
    ):
        self._relaxation = relaxation
        self._model = model
        self._primal = primal

    @property
    def value(self) -> np.float64 | np.complex128:
        return self._model.primalObjValue()

    def __getitem__(self, monomial) -> np.float64 | np.complex128:
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
            sign = 1 if self._model.getTask().getobjsense() == "maximize" else -1

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
    def moment_matrix(self) -> np.ndarray:
        size = self._relaxation.moment_matrices[0].size

        if self._primal:
            moment_matrix_level = self._model.getConstraint("MM-0").level()
        else:
            moment_matrix_level = self._model.getVariable("Y_0").dual()

        if self._relaxation.is_real:
            return moment_matrix_level.reshape(size, size)

        moment_matrix_level = moment_matrix_level.reshape(2 * size, 2 * size)

        return moment_matrix_level[:size, :size] + 1j * moment_matrix_level[size:, :size]

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

            moment_matrix_level = moment_matrix_level.reshape(2 * size, 2 * size)
            res[id] = moment_matrix_level[:size, :size] + 1j * moment_matrix_level[size:, :size]

        return res

    @property
    def localizing_matrices_equality_constraints(self) -> list[np.ndarray]:
        res = []

        for index, localizing_moment_matrix in enumerate(self._relaxation.localising_moment_matrices_equalities[0]):
            if self._primal:
                localizing_moment_matrix_level = self._model.getConstraint(f"LMME-0-{index}").level()
            else:
                localizing_moment_matrix_level = (
                    self._model.getVariable(f"Q_(0, {index})^0").dual()
                    - self._model.getVariable(f"Q_(0, {index})^1").dual()
                )

            if self._relaxation.is_real:
                res.append(
                    localizing_moment_matrix_level.reshape(localizing_moment_matrix.size, localizing_moment_matrix.size)
                )
            else:
                localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                    2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                )
                res.append(
                    localizing_moment_matrix_level[: localizing_moment_matrix.size, : localizing_moment_matrix.size]
                    + 1j
                    * localizing_moment_matrix_level[localizing_moment_matrix.size :, : localizing_moment_matrix.size]
                )

        return res

    @property
    def localizing_matrices_inequality_constraints(self) -> list[np.ndarray]:
        res = []

        for index, localizing_moment_matrix in enumerate(self._relaxation.localising_moment_matrices_inequalities[0]):
            if self._primal:
                localizing_moment_matrix_level = self._model.getConstraint(f"LMMI-0-{index}").level()
            else:
                localizing_moment_matrix_level = self._model.getVariable(f"P_(0, {index})").dual()

            if self._relaxation.is_real:
                res.append(
                    localizing_moment_matrix_level.reshape(localizing_moment_matrix.size, localizing_moment_matrix.size)
                )
            else:
                localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                    2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                )
                res.append(
                    localizing_moment_matrix_level[: localizing_moment_matrix.size, : localizing_moment_matrix.size]
                    + 1j
                    * localizing_moment_matrix_level[localizing_moment_matrix.size :, : localizing_moment_matrix.size]
                )

        return res

    @property
    def localizing_matrices_equality_constraints_by_mm_id(self) -> dict[int, list[np.ndarray]]:
        res = {}

        for (
            id,
            localizing_moment_matrices_equalities_id,
        ) in self._relaxation.localising_moment_matrices_equalities.items():
            to_add = []

            for index, localizing_moment_matrix in enumerate(localizing_moment_matrices_equalities_id):
                if self._primal:
                    localizing_moment_matrix_level = self._model.getConstraint(f"LMME-{id}-{index}").level()
                else:
                    localizing_moment_matrix_level = (
                        self._model.getVariable(f"Q_({id}, {index})^0").dual()
                        - self._model.getVariable(f"Q_({id}, {index})^1").dual()
                    )

                if self._relaxation.is_real:
                    to_add.append(
                        localizing_moment_matrix_level.reshape(
                            localizing_moment_matrix.size, localizing_moment_matrix.size
                        )
                    )
                else:
                    localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_add.append(
                        localizing_moment_matrix_level[: localizing_moment_matrix.size, : localizing_moment_matrix.size]
                        + 1j
                        * localizing_moment_matrix_level[
                            localizing_moment_matrix.size :, : localizing_moment_matrix.size
                        ]
                    )

            res[id] = to_add

        return res

    @property
    def localizing_matrices_inequality_constraints_by_mm_id(self) -> dict[int, list[np.ndarray]]:
        res = {}

        for (
            id,
            localizing_moment_matrices_inequalities_id,
        ) in self._relaxation.localising_moment_matrices_inequalities.items():
            to_add = []

            for index, localizing_moment_matrix in enumerate(localizing_moment_matrices_inequalities_id):
                if self._primal:
                    localizing_moment_matrix_level = self._model.getConstraint(f"LMMI-{id}-{index}").level()
                else:
                    localizing_moment_matrix_level = self._model.getVariable(f"P_({id}, {index})").dual()

                if self._relaxation.is_real:
                    to_add.append(
                        localizing_moment_matrix_level.reshape(
                            localizing_moment_matrix.size, localizing_moment_matrix.size
                        )
                    )
                else:
                    localizing_moment_matrix_level = localizing_moment_matrix_level.reshape(
                        2 * localizing_moment_matrix.size, 2 * localizing_moment_matrix.size
                    )
                    to_add.append(
                        localizing_moment_matrix_level[: localizing_moment_matrix.size, : localizing_moment_matrix.size]
                        + 1j
                        * localizing_moment_matrix_level[
                            localizing_moment_matrix.size :, : localizing_moment_matrix.size
                        ]
                    )

            res[id] = to_add

        return res
