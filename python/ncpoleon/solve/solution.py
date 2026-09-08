from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Generic, cast

import numpy as np

from ncpoleon._typing import MonomialType, RealOrComplexMatrix, Scalar
from ncpoleon.relaxations import Hermiticity
from ncpoleon.solve.utils import sos_vectors_of_hermitian_matrix

from .sos_decomposition import (
    LocalizingMomentMatrixHermitianEqualityDecomposition,
    LocalizingMomentMatrixInequalityDecomposition,
    LocalizingMomentMatrixNonHermitianEqualityDecomposition,
    MomentMatrixDecomposition,
    SingleMomentEqualityDecomposition,
    SingleMomentInequalityDecomposition,
    SoSDecomposition,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation


class BaseSolution(ABC, Generic[MonomialType, Scalar]):
    @property
    @abstractmethod
    def value(self) -> float: ...

    @abstractmethod
    def __getitem__(self, monomial: MonomialType) -> Scalar: ...

    @property
    def moment_matrix(self) -> RealOrComplexMatrix:
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
    ) -> dict[int, RealOrComplexMatrix]: ...

    @property
    def moment_matrix_multiplier(self) -> RealOrComplexMatrix:
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
    ) -> dict[int, RealOrComplexMatrix]: ...

    @property
    def localizing_matrices_hermitian_equality_multipliers(
        self,
    ) -> list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]]:
        localizing_matrices_multipliers = self.localizing_matrices_hermitian_equality_multipliers_by_mm_id
        if len(localizing_matrices_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_equality_multipliers` "
                "property will only return the equality localizing moment matrices multipliers associated to the moment"
                " matrix of index 0. Use `localizing_matrices_equality_multipliers_by_mm_id` to access all of them.",
            )
        return localizing_matrices_multipliers[0]

    @abstractmethod
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
        """The multiplier of every moment equality a localizing equality was expanded into, by moment matrix id."""

    @property
    def localizing_matrices_hermitian_equality_multipliers_by_mm_id(
        self,
    ) -> dict[int, list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]]]:
        res: dict[int, list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]]] = {}

        for mm_id, list_of_multipliers in self._localizing_matrices_equality_multipliers_by_mm_id(
            Hermiticity.Hermitian
        ).items():
            res_id: list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]] = []

            for generator, moments_with_multipliers, generating_set in list_of_multipliers:
                matrix_size = len(generating_set)
                dtype = complex if any(np.iscomplexobj(m) for _moment, m in moments_with_multipliers) else float
                matrix = np.empty((matrix_size, matrix_size), dtype=dtype)

                # A hermitian generator only yields the upper triangle of its localizing matrix, in row-major
                # order, so the flat list walks (0, 0), (0, 1), ..., (1, 1), ... rather than a full square
                for (index_row, index_col), (_moment, multiplier) in zip(
                    combinations_with_replacement(range(matrix_size), 2), moments_with_multipliers, strict=True
                ):
                    if index_row == index_col:
                        matrix[index_row, index_col] = multiplier
                    else:
                        matrix[index_row, index_col] = multiplier / 2
                        matrix[index_col, index_row] = np.conj(multiplier) / 2

                res_id.append((generator, matrix, generating_set))

            res[mm_id] = res_id

        return res

    @property
    def localizing_matrices_nonhermitian_equality_multipliers(
        self,
    ) -> list[tuple[Polynomial[MonomialType, Scalar], Sequence[tuple[Polynomial[MonomialType, Scalar], Scalar]]]]:
        localizing_matrices_multipliers = self.localizing_matrices_nonhermitian_equality_multipliers_by_mm_id
        if len(localizing_matrices_multipliers) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `localizing_matrices_equality_multipliers` "
                "property will only return the equality localizing moment matrices multipliers associated to the moment"
                " matrix of index 0. Use `localizing_matrices_equality_multipliers_by_mm_id` to access all of them.",
            )
        return localizing_matrices_multipliers[0]

    @property
    def localizing_matrices_nonhermitian_equality_multipliers_by_mm_id(
        self,
    ) -> dict[
        int,
        list[tuple[Polynomial[MonomialType, Scalar], Sequence[tuple[Polynomial[MonomialType, Scalar], Scalar]]]],
    ]:
        return {
            mm_id: [(generator, moments) for (generator, moments, _generating_set) in res_id]
            for mm_id, res_id in self._localizing_matrices_equality_multipliers_by_mm_id(
                Hermiticity.NonHermitian
            ).items()
        }

    @property
    def localizing_matrices_inequality(
        self,
    ) -> list[
        tuple[
            Polynomial[MonomialType, Scalar],
            RealOrComplexMatrix,
            list[MonomialType],
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
                Polynomial[MonomialType, Scalar],
                RealOrComplexMatrix,
                list[MonomialType],
            ]
        ],
    ]: ...

    @property
    def localizing_matrices_inequality_multipliers(
        self,
    ) -> list[
        tuple[
            Polynomial[MonomialType, Scalar],
            RealOrComplexMatrix,
            list[MonomialType],
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
                Polynomial[MonomialType, Scalar],
                RealOrComplexMatrix,
                list[MonomialType],
            ]
        ],
    ]: ...

    @property
    @abstractmethod
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[MonomialType, Scalar], float | complex]]: ...

    @property
    @abstractmethod
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[MonomialType, Scalar], float]]: ...

    @property
    @abstractmethod
    def relaxation(self) -> BaseSdpRelaxation[MonomialType, Scalar]: ...

    def get_sos_decomposition(
        self, *, cutoff: float = 0.0, delta: float = 0.0
    ) -> SoSDecomposition[MonomialType, Scalar]:
        sos_decompositions = self.get_sos_decomposition_by_mm_id(cutoff=cutoff, delta=delta)
        if len(sos_decompositions) > 1:
            warnings.warn(
                "The solution contains multiple moment matrices. The `get_sos_decomposition` "
                "method will only return the SoS decomposition associated to the moment matrix of"
                " index 0. Use `get_sos_decomposition_by_mm_id` to access all of them.",
            )
        return sos_decompositions[0]

    def get_sos_decomposition_by_mm_id(
        self, *, cutoff: float = 0.0, delta: float = 0.0
    ) -> dict[int, SoSDecomposition[MonomialType, Scalar]]:
        res: dict[int, SoSDecomposition[MonomialType, Scalar]] = {}
        moment_matrix_multipliers = self.moment_matrix_multiplier_by_mm_id
        localizing_moment_matrices_multipliers_nonhermitian_equality = (
            self.localizing_matrices_nonhermitian_equality_multipliers_by_mm_id
        )
        localizing_moment_matrices_multipliers_hermitian_equality = (
            self.localizing_matrices_hermitian_equality_multipliers_by_mm_id
        )
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
            n_monomials = sos_vectors.shape[0]
            decomposition = (np.array(self.relaxation.generating_sets[mm_id][:n_monomials]) @ sos_vectors).tolist()

            if delta:
                decomposition = [self.relaxation.rewrite(p).chop(delta) for p in decomposition]
                decomposition = [p for p in decomposition if not p.is_zero()]

            moment_matrix_term = MomentMatrixDecomposition(decomposition=decomposition)

            inequalities_terms = []

            for generator, coefficient, generating_set in localizing_moment_matrices_multipliers_inequality.get(
                mm_id, []
            ):
                sos_vectors = sos_vectors_of_hermitian_matrix(coefficient, cutoff)[0]
                decompositions = (np.array(generating_set) @ sos_vectors).tolist()

                if delta:
                    decompositions = [self.relaxation.rewrite(p).chop(delta) for p in decompositions]
                    decompositions = [p for p in decompositions if not p.is_zero()]

                if decompositions:
                    inequalities_terms.append(
                        LocalizingMomentMatrixInequalityDecomposition(generator=generator, decomposition=decompositions)
                    )

            non_hermitian_equalities_terms = []

            for (
                _generator,
                moments_with_multipliers,
            ) in localizing_moment_matrices_multipliers_nonhermitian_equality.get(mm_id, []):
                to_hermitianize = [
                    cast("Polynomial[MonomialType, Scalar]", multiplier * moment.adjoint())
                    for moment, multiplier in moments_with_multipliers
                ]
                term = sum([(poly + poly.adjoint()) / 2 for poly in to_hermitianize])

                if not isinstance(term, int):  # sum returns 0 on empty lists
                    term = self.relaxation.rewrite(term).chop(delta)

                    if not term.is_zero():
                        non_hermitian_equalities_terms.append(
                            LocalizingMomentMatrixNonHermitianEqualityDecomposition(term=term)
                        )

            hermitian_equalities_terms = []

            for generator, coefficient, generating_set in localizing_moment_matrices_multipliers_hermitian_equality.get(
                mm_id, []
            ):
                sos_vectors_pos, sos_vectors_neg = sos_vectors_of_hermitian_matrix(coefficient, cutoff)
                decomposition_positive = (np.array(generating_set) @ sos_vectors_pos).reshape(-1).tolist()
                decomposition_negative = (np.array(generating_set) @ sos_vectors_neg).reshape(-1).tolist()
                hermitian_equalities_terms.append(
                    LocalizingMomentMatrixHermitianEqualityDecomposition(
                        generator=generator,
                        decomposition_positive=decomposition_positive,
                        decomposition_negative=decomposition_negative,
                    )
                )

            # Annotated because the branches below append differently-specialized instances, and a
            # bare `[]` would infer a union element type that the invariant `list` then rejects
            moment_equalities_terms: list[SingleMomentEqualityDecomposition[MonomialType, Scalar]] = []

            for generator, coefficient in moment_equality_multipliers.get(mm_id, []):
                to_hermitianize = coefficient * generator.adjoint()
                to_add = self.relaxation.rewrite((to_hermitianize + to_hermitianize.adjoint()) / 2).chop(delta)

                if not to_add.is_zero():
                    moment_equalities_terms.append(
                        SingleMomentEqualityDecomposition(
                            term=to_add,
                        )
                    )

            moment_inequalities_terms = []

            for generator, coefficient in moment_inequality_multipliers.get(mm_id, []):
                to_add = self.relaxation.rewrite(coefficient * generator).chop(delta)

                if not to_add.is_zero():
                    moment_inequalities_terms.append(SingleMomentInequalityDecomposition(term=to_add))

            res[mm_id] = SoSDecomposition[MonomialType, Scalar](
                moment_matrix_term=moment_matrix_term,
                nonhermitian_equalities_terms=non_hermitian_equalities_terms,
                hermitian_equalities_terms=hermitian_equalities_terms,
                inequalities_terms=inequalities_terms,
                moment_equalities_terms=moment_equalities_terms,
                moment_inequalities_terms=moment_inequalities_terms,
            )

        return res
