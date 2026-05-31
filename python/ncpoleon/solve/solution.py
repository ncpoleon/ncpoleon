from abc import ABC, abstractmethod

import numpy as np

class BaseSolution(ABC):
    @property
    @abstractmethod
    def value(self) -> np.float64 | np.complex128:
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

    @property
    @abstractmethod
    def moment_equalities_coefficients(self) -> list[np.float64 | np.complex128]:
        pass

    @property
    @abstractmethod
    def moment_inequalities_coefficients(self) -> list[np.float64]:
        pass

    def get_sos_decomposition(self):
        pass
