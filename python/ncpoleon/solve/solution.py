from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mosek.fusion import Model

    from ncpoleon.relaxations import (
        ComplexValuedCommutativeSdpRelaxation,
        ComplexValuedNonCommutativeSdpRelaxation,
        RealValuedCommutativeSdpRelaxation,
        RealValuedNonCommutativeSdpRelaxation,
    )


class BaseSolution(ABC):
    @abstractmethod
    def get_sos_decomposition(self):
        pass

    @abstractmethod
    def __getitem__(self, monomial) -> np.float64 | np.complex128:
        pass

    @property
    @abstractmethod
    def moment_matrix(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def localizing_matrices_equality_constraints(self) -> list[np.ndarray]:
        pass

    @property
    @abstractmethod
    def localizing_matrices_inequality_constraints(self) -> list[np.ndarray]:
        pass

    @property
    @abstractmethod
    def moment_matrix_by_mm_id(self) -> dict[int, np.ndarray]:
        pass

    @property
    @abstractmethod
    def localizing_matrices_equality_constraints_by_mm_id(self) -> dict[int, list[np.ndarray]]:
        pass

    @property
    @abstractmethod
    def localizing_matrices_inequality_constraints_by_mm_id(self) -> dict[int, list[np.ndarray]]:
        pass


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
        raise NotImplementedError("Can't access a primal value using the dual for now")

    @property
    def moment_matrix(self):
        mm = self._relaxation.moment_matrices[0]

        return self._model.getConstraint("MM-0").level().reshape(mm.size, mm.size)

    @property
    def moment_matrix_by_mm_id(self):
        return {
            id: self._model.getConstraint(f"MM-{id}").level().reshape(moment_matrix.size, moment_matrix.size)
            for id, moment_matrix in self._relaxation.moment_matrices.items()
        }

    @property
    def localizing_matrices_equality_constraints(self) -> list[np.ndarray]:
        return [
            self._model.getConstraint(f"LMME-{0}-{index}")
            .level()
            .reshape(localizing_moment_matrix.size, localizing_moment_matrix.size)
            for index, localizing_moment_matrix in enumerate(self._relaxation.localising_moment_matrices_equalities[0])
        ]

    @property
    def localizing_matrices_inequality_constraints(self) -> list[np.ndarray]:
        return [
            self._model.getConstraint(f"LMMI-{0}-{index}")
            .level()
            .reshape(localizing_moment_matrix.size, localizing_moment_matrix.size)
            for index, localizing_moment_matrix in enumerate(
                self._relaxation.localising_moment_matrices_inequalities[0]
            )
        ]

    @property
    def localizing_matrices_equality_constraints_by_mm_id(self) -> dict[int, list[np.ndarray]]:
        return {
            id: [
                self._model.getConstraint(f"LMME-{id}-{index}")
                .level()
                .reshape(localizing_moment_matrix.size, localizing_moment_matrix.size)
                for index, localizing_moment_matrix in enumerate(
                    self._relaxation.localising_moment_matrices_equalities[id]
                )
            ]
            for id in self._relaxation.moment_matrices
        }

    @property
    def localizing_matrices_inequality_constraints_by_mm_id(self) -> dict[int, list[np.ndarray]]:
        return {
            id: [
                self._model.getConstraint(f"LMMI-{id}-{index}")
                .level()
                .reshape(localizing_moment_matrix.size, localizing_moment_matrix.size)
                for index, localizing_moment_matrix in enumerate(
                    self._relaxation.localising_moment_matrices_inequalities[id]
                )
            ]
            for id in self._relaxation.moment_matrices
        }

    def get_sos_decomposition(self):
        if self._primal:
            raise NotImplementedError("Can't derive the SOS decomposition by solving the primal for now")
