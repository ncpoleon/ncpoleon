from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

import numpy as np

from ncpoleon._typing import PolynomialElements, Scalar
from ncpoleon.solve.utils import sos_vectors_of_hermitian_matrix

from .sos_decomposition import (
    LocalizingMomentMatrixEqualityDecomposition,
    LocalizingMomentMatrixInequalityDecomposition,
    MomentMatrixDecomposition,
    SingleMomentEqualityDecomposition,
    SingleMomentInequalityDecomposition,
    SoSDecomposition,
)

if TYPE_CHECKING:
    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation


class BaseSolution(ABC, Generic[PolynomialElements, Scalar]):
    @property
    @abstractmethod
    def value(self) -> float: ...

    @abstractmethod
    def __getitem__(self, monomial: PolynomialElements) -> Scalar: ...

    @property
    def moment_matrix(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]:
        moment_matrices = self.moment_matrix_by_mm_id
        if len(moment_matrices) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `moment_matrix` property will only return the one "
                "associated to index 0. Use `moment_matrix_by_mm_id` to access all of them.",
            )
        return moment_matrices[0]

    @property
    @abstractmethod
    def moment_matrix_by_mm_id(
        self,
    ) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]]: ...

    @property
    def moment_matrix_multiplier(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]:
        moment_matrix_multipliers = self.moment_matrix_multiplier_by_mm_id
        if len(moment_matrix_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `moment_matrix_multiplier` property will only "
                "return the one associated to index 0. Use `moment_matrix_multiplier_by_mm_id` to access all of them.",
            )
        return moment_matrix_multipliers[0]

    @property
    @abstractmethod
    def moment_matrix_multiplier_by_mm_id(
        self,
    ) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]]: ...

    @property
    def localizing_matrices_equality_multipliers(
        self,
    ) -> list[
        tuple[
            Polynomial[PolynomialElements, Scalar],
            np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]],
        ]
    ]:
        localizing_matrices_multipliers = self.localizing_matrices_equality_multipliers_by_mm_id
        if len(localizing_matrices_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_equality_multipliers` "
                "property will only return the equality localizing moment matrices multipliers associated to the moment"
                " matrix of index 0. Use `localizing_matrices_equality_multipliers_by_mm_id` to access all of them.",
            )
        return localizing_matrices_multipliers[0]

    @property
    @abstractmethod
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
    ]: ...

    @property
    def localizing_matrices_inequality(
        self,
    ) -> list[
        tuple[
            Polynomial[PolynomialElements, Scalar],
            np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]],
        ]
    ]:
        localizing_matrices = self.localizing_matrices_inequality_by_mm_id
        if len(localizing_matrices) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_inequality` "
                "property will only return the inequality localizing moment matrices associated to the moment matrix of"
                " index 0. Use `localizing_matrices_inequality_by_mm_id` to access all of them.",
            )
        return localizing_matrices[0]

    @property
    @abstractmethod
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
    ]: ...

    @property
    def localizing_matrices_inequality_multipliers(
        self,
    ) -> list[
        tuple[
            Polynomial[PolynomialElements, Scalar],
            np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]],
        ]
    ]:
        localizing_matrices_multipliers = self.localizing_matrices_inequality_multipliers_by_mm_id
        if len(localizing_matrices_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_inequality_multipliers` "
                "property will only return the inequality localizing moment matrices multipliers associated to the "
                "moment matrix of index 0. Use `localizing_matrices_inequality_multipliers_by_mm_id` to access all of "
                "them.",
            )
        return localizing_matrices_multipliers[0]

    @property
    @abstractmethod
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
    ]: ...

    @property
    @abstractmethod
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], float | complex]]: ...

    @property
    @abstractmethod
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], float]]: ...

    @property
    @abstractmethod
    def relaxation(self) -> BaseSdpRelaxation[PolynomialElements, Scalar]: ...

    def get_sos_decomposition(self, *, cutoff: float = 0.0) -> SoSDecomposition[PolynomialElements, Scalar]:
        sos_decompositions = self.get_sos_decomposition_by_mm_id(cutoff=cutoff)
        if len(sos_decompositions) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `get_sos_decomposition` "
                "method will only return the SoS decomposition associated to the moment matrix of"
                " index 0. Use `get_sos_decomposition_by_mm_id` to access all of them.",
            )
        return sos_decompositions[0]

    def get_sos_decomposition_by_mm_id(
        self, *, cutoff: float = 0.0
    ) -> dict[int, SoSDecomposition[PolynomialElements, Scalar]]:
        res: dict[int, SoSDecomposition[PolynomialElements, Scalar]] = {}
        moment_matrix_multipliers = self.moment_matrix_multiplier_by_mm_id
        localizing_moment_matrices_multipliers_equality = self.localizing_matrices_equality_multipliers_by_mm_id
        localizing_moment_matrices_multipliers_inequality = self.localizing_matrices_inequality_multipliers_by_mm_id
        moment_equality_multipliers = {}

        for polynomial, scalar in self.moment_equalities_multipliers:
            for mm_id, poly_id in polynomial.by_moment_matrix_id().items():
                if mm_id in moment_equality_multipliers:
                    moment_equality_multipliers[mm_id].append((poly_id, scalar))
                else:
                    moment_equality_multipliers[mm_id] = [(poly_id, scalar)]

        moment_inequality_multipliers = {}

        for polynomial, scalar in self.moment_inequalities_multipliers:
            for mm_id, poly_id in polynomial.by_moment_matrix_id().items():
                if mm_id in moment_inequality_multipliers:
                    moment_inequality_multipliers[mm_id].append((poly_id, scalar))
                else:
                    moment_inequality_multipliers[mm_id] = [(poly_id, scalar)]

        for mm_id in self.relaxation.moment_matrices:
            sos_vectors = sos_vectors_of_hermitian_matrix(moment_matrix_multipliers[mm_id], cutoff)[0]
            n_monomials = sos_vectors.shape[1]
            decomposition = (sos_vectors @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
            moment_matrix_term = MomentMatrixDecomposition(decomposition=decomposition)

            inequalities_terms = []

            for generator, coefficient in localizing_moment_matrices_multipliers_inequality.get(mm_id, []):
                sos_vectors = sos_vectors_of_hermitian_matrix(coefficient, cutoff)[0]
                n_monomials = sos_vectors.shape[1]
                decompositions = (
                    (sos_vectors @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
                )
                inequalities_terms.append(
                    LocalizingMomentMatrixInequalityDecomposition(generator=generator, decomposition=decompositions)
                )

            equalities_terms = []

            for generator, coefficient in localizing_moment_matrices_multipliers_equality.get(mm_id, []):
                sos_vectors_pos, sos_vectors_neg = sos_vectors_of_hermitian_matrix(coefficient, cutoff)
                n_monomials = sos_vectors_pos.shape[1]
                decomposition_positive = (
                    (sos_vectors_pos @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
                )
                decomposition_negative = (
                    (sos_vectors_neg @ self.relaxation.generating_sets[mm_id][:n_monomials]).reshape(-1).tolist()
                )
                equalities_terms.append(
                    LocalizingMomentMatrixEqualityDecomposition(
                        generator=generator,
                        decomposition_positive=decomposition_positive,
                        decomposition_negative=decomposition_negative,
                    )
                )

            moment_equalities_terms = []

            for generator, coefficient in moment_equality_multipliers.get(mm_id, []):
                if isinstance(coefficient, float):
                    moment_equalities_terms.append(
                        SingleMomentEqualityDecomposition(
                            generator=(self.relaxation.rewrite(generator + generator.adjoint()) / 2),
                            coefficient=coefficient,
                        )
                    )
                elif isinstance(coefficient, complex):
                    moment_equalities_terms.append(
                        SingleMomentEqualityDecomposition(
                            generator=(self.relaxation.rewrite(generator + generator.adjoint()) / 2),
                            coefficient=coefficient.real,
                        )
                    )
                    moment_equalities_terms.append(
                        SingleMomentEqualityDecomposition(
                            generator=(self.relaxation.rewrite(generator.adjoint() - generator) / 2),
                            coefficient=coefficient.imag * 1j,
                        )
                    )
                else:
                    raise ValueError(
                        f"Lagrange multiplier for moment equality {generator} has type {type(coefficient)}, but only "
                        "floats and complex are supported."
                    )

            moment_inequalities_terms = []

            for generator, coefficient in moment_inequality_multipliers.get(mm_id, []):
                moment_inequalities_terms.append(
                    SingleMomentInequalityDecomposition(generator=generator, coefficient=coefficient)
                )

            res[mm_id] = SoSDecomposition[PolynomialElements, Scalar](
                moment_matrix_term=moment_matrix_term,
                equalities_terms=equalities_terms,
                inequalities_terms=inequalities_terms,
                moment_equalities_terms=moment_equalities_terms,
                moment_inequalities_terms=moment_inequalities_terms,
            )

        return res
