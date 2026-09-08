from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ncpoleon._typing import MonomialType, RealOrComplexMatrix, Scalar
from ncpoleon.relaxations import Canonicality, Hermiticity, Realness
from ncpoleon.solve.solution import BaseSolution
from ncpoleon.utils import is_vacuous_moment

if TYPE_CHECKING:
    import picos as pc

    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation


class PicosSolution(BaseSolution[MonomialType, Scalar]):
    def __init__(
        self,
        relaxation: BaseSdpRelaxation[MonomialType, Scalar],
        problem: pc.Problem,
        constraints: dict[str, pc.constraints.Constraint],
        psd_matrices: dict[str, pc.expressions.Expression],
        primal: bool,
    ):
        self._relaxation = relaxation
        self._problem = problem
        self._primal = primal
        self._constraints = constraints
        self._psd_matrices = psd_matrices

    @property
    def value(self) -> float:
        return self._problem.value

    @property
    def relaxation(self) -> BaseSdpRelaxation[MonomialType, Scalar]:
        return self._relaxation

    def __getitem__(self, monomial: MonomialType) -> Scalar:
        rewritten_monomial = self._relaxation.rewrite(monomial)
        canonical_monomial, canonicality, realness = self._relaxation.moment_matrices[
            rewritten_monomial.moment_matrix_id
        ].get_canonical(rewritten_monomial)

        if self._primal:
            if realness == Realness.Real or canonicality == Canonicality.Canonical:
                return cast(float, self._problem.get_variable(str(canonical_monomial)).value)
            return cast(Scalar, self._problem.get_variable(str(canonical_monomial)).value.conjugate())
        else:
            dual = -self._constraints[f"M-{canonical_monomial}"].dual

            if realness == Realness.Real:
                return cast(float, dual)

            # A complex monomial and its adjoint share one constraint, whose row is half the Lagrangian pairing,
            # and PICOS reports the multiplier of a complex equality conjugated
            dual /= 2
            return cast(Scalar, dual.conjugate()) if canonicality == Canonicality.Canonical else cast(Scalar, dual)

    @property
    def moment_matrix_by_mm_id(
        self,
    ) -> dict[int, RealOrComplexMatrix]:
        res: dict[int, RealOrComplexMatrix] = {}

        for id in self._relaxation.moment_matrices:
            if self._primal:
                res[id] = np.array(self._psd_matrices[f"MM-{id}"].value)
            else:
                res[id] = np.array(self._constraints[f"Y_{id}"].dual).conj()

            if not res[id].shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                res[id] = res[id].reshape((1, 1))

        return res

    @property
    def moment_matrix_multiplier_by_mm_id(
        self,
    ) -> dict[int, RealOrComplexMatrix]:
        res: dict[int, RealOrComplexMatrix] = {}

        for id in self._relaxation.moment_matrices:
            if self._primal:
                res[id] = np.array(self._constraints[f"MM-{id}"].dual).conj()
            else:
                res[id] = np.array(self._problem.get_variable(f"Y_{id}").value)

            if not res[id].shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                res[id] = res[id].reshape((1, 1))

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
            ) in enumerate(
                zip(equalities_as_moments, self._relaxation.equalities.get(moment_matrix_id, []), strict=True)
            ):
                if equality_hermiticity != hermiticity:
                    continue

                list_of_moments: list[tuple[Polynomial[MonomialType, Scalar], Scalar]] = []

                for moment_index, moment in enumerate(equality_as_moments):
                    # A vacuous moment got neither a constraint nor a variable, and 0 satisfies it
                    if is_vacuous_moment(self._relaxation, moment):
                        list_of_moments.append((moment, cast("Scalar", 0.0)))
                        continue

                    if self._primal:
                        multiplier = self._constraints[f"ME-{moment_matrix_id}-{equality_index}-{moment_index}"].dual
                    else:
                        multiplier = self._problem.get_variable(
                            f"nu_{(moment_matrix_id, equality_index, moment_index)}"
                        ).value

                    list_of_moments.append((moment, cast("Scalar", multiplier)))

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

        for id in self._relaxation.localising_moment_matrices_inequalities:
            to_add: list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]] = []

            for index, (inequality_constraint, generating_set) in enumerate(self._relaxation.inequalities.get(id, [])):
                if self._primal:
                    to_append = np.array(self._psd_matrices[f"LMMI-{id}-{index}"].value)
                else:
                    to_append = np.array(self._constraints[f"P_({id}, {index})"].dual).conj()

                if not to_append.shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                    to_append = to_append.reshape((1, 1))

                to_add.append((inequality_constraint, to_append, generating_set))

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

        for id in self._relaxation.localising_moment_matrices_inequalities:
            to_add: list[tuple[Polynomial[MonomialType, Scalar], RealOrComplexMatrix, list[MonomialType]]] = []

            for index, (inequality_constraint, generating_set) in enumerate(self._relaxation.inequalities.get(id, [])):
                if self._primal:
                    to_append = np.array(self._constraints[f"LMMI-{id}-{index}"].dual).conj()
                else:
                    to_append = np.array(self._problem.get_variable(f"P_({id}, {index})").value)

                if not to_append.shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                    to_append = to_append.reshape((1, 1))

                to_add.append((inequality_constraint, to_append, generating_set))

            res[id] = to_add

        return res

    @property
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[MonomialType, Scalar], np.float64 | np.complex128]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_equalities):
            if self._primal:
                res.append((polynomial_constraint, self._constraints[f"ME-{index}"].dual))
            else:
                res.append((polynomial_constraint, self._problem.get_variable(f"nu_{index}").value))

        return res

    @property
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[MonomialType, Scalar], np.float64]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_inequalities):
            if self._primal:
                res.append((polynomial_constraint, self._constraints[f"MI-{index}"].dual))
            else:
                res.append((polynomial_constraint, self._problem.get_variable(f"lambda_{index}").value))

        return res
