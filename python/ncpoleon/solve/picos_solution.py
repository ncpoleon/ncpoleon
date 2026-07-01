from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ncpoleon._typing import PolynomialElements, Scalar
from ncpoleon.solve.solution import BaseSolution

if TYPE_CHECKING:
    import picos as pc

    from ncpoleon.polynomials import Polynomial
    from ncpoleon.relaxations import BaseSdpRelaxation


class PicosSolution(BaseSolution[PolynomialElements, Scalar]):
    def __init__(
        self,
        relaxation: BaseSdpRelaxation[PolynomialElements, Scalar],
        problem: pc.Problem,
        constraints: dict[str, pc.Constraint],
        primal: bool,
    ):
        self._relaxation = relaxation
        self._problem = problem
        self._primal = primal
        self._constraints = constraints

    @property
    def value(self) -> float:
        return self._problem.value

    @property
    def relaxation(self) -> BaseSdpRelaxation[PolynomialElements, Scalar]:
        return self._relaxation

    def __getitem__(self, monomial) -> Scalar:
        rewritten_monomial = self._relaxation.rewrite(monomial)
        canonical_monomial, is_adjoint, is_real_valued = self._relaxation.moment_matrices[
            rewritten_monomial.moment_matrix_id
        ].get_canonical(rewritten_monomial)

        if self._primal:
            if is_adjoint and not is_real_valued:
                return self._problem.get_variable(str(canonical_monomial)).value.conjugate()
            else:
                return self._problem.get_variable(str(canonical_monomial)).value
        else:
            return -self._problem.get_constraint(self._constraints[f"M-{canonical_monomial}"]).dual

    @property
    def moment_matrix_by_mm_id(
        self,
    ) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]]:
        res = {}

        for id in self._relaxation.moment_matrices:
            if self._primal:
                res[id] = np.array(self._problem.get_constraint(self._constraints[f"MM-{id}"]).lhs.value)
            else:
                res[id] = np.array(self._problem.get_constraint(self._constraints[f"Y_{id}"]).dual)

            if not res[id].shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                res[id] = res[id].reshape((1, 1))

        return res

    @property
    def moment_matrix_multiplier_by_mm_id(
        self,
    ) -> dict[int, np.ndarray[tuple[int, int], np.dtype[np.float64] | np.dtype[np.complex128]]]:
        res = {}

        for id in self._relaxation.moment_matrices:
            if self._primal:
                res[id] = np.array(self._problem.get_constraint(self._constraints[f"MM-{id}"]).dual)
            else:
                res[id] = np.array(self._problem.get_variable(f"Y_{id}").value)

            if not res[id].shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                res[id] = res[id].reshape((1, 1))

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

        for id in self._relaxation.localising_moment_matrices_equalities:
            to_add = []

            for index, equality_constraint in enumerate(self._relaxation.equalities.get(id, [])):
                # The equality constraints on symmetric matrices are redundant, and thus Picos doesn't return a
                # Hermitian matrix for the dual, so we have to hermitianize it
                if self._primal:
                    to_hermitianize = np.array(
                        self._problem.get_constraint(self._constraints[f"LMME-{id}-{index}"]).dual
                    )
                    to_append = (to_hermitianize + to_hermitianize.T.conj()) / 2
                else:
                    to_append = np.array(self._problem.get_variable(f"Q_{(id, index)}").value)

                if not to_append.shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                    to_append = to_append.reshape((1, 1))

                to_add.append((equality_constraint, to_append))

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

        for id in self._relaxation.localising_moment_matrices_inequalities:
            to_add = []

            for index, inequality_constraint in enumerate(self._relaxation.inequalities.get(id, [])):
                if self._primal:
                    to_append = np.array(
                        self._problem.get_constraint(self._constraints[f"LMMI-{id}-{index}"]).lhs.value
                    )
                else:
                    to_append = np.array(self._problem.get_constraint(self._constraints[f"P_({id}, {index})"]).dual)

                if not to_append.shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                    to_append = to_append.reshape((1, 1))

                to_add.append((inequality_constraint, to_append))

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

        for id in self._relaxation.localising_moment_matrices_inequalities:
            to_add = []

            for index, inequality_constraint in enumerate(self._relaxation.inequalities.get(id, [])):
                if self._primal:
                    to_append = np.array(self._problem.get_constraint(self._constraints[f"LMMI-{id}-{index}"]).dual)
                else:
                    to_append = np.array(self._problem.get_variable(f"P_({id}, {index})").value)

                if not to_append.shape:  # For 1x1 constraints or variables, Picos returns a 0D array
                    to_append = to_append.reshape((1, 1))

                to_add.append((inequality_constraint, to_append))

            res[id] = to_add

        return res

    @property
    def moment_equalities_multipliers(
        self,
    ) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.float64 | np.complex128]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_equalities):
            if self._primal:
                res.append((polynomial_constraint, self._problem.get_constraint(self._constraints[f"ME-{index}"]).dual))
            else:
                res.append((polynomial_constraint, self._problem.get_variable(f"nu_{index}").value))

        return res

    @property
    def moment_inequalities_multipliers(self) -> list[tuple[Polynomial[PolynomialElements, Scalar], np.float64]]:
        res = []

        for index, (polynomial_constraint, _scalar) in enumerate(self._relaxation.moment_inequalities):
            if self._primal:
                res.append((polynomial_constraint, self._problem.get_constraint(self._constraints[f"MI-{index}"]).dual))
            else:
                res.append((polynomial_constraint, self._problem.get_variable(f"lambda_{index}").value))

        return res
