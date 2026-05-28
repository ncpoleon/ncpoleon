from abc import ABC, abstractmethod


class BaseSolution(ABC):
    @abstractmethod
    def get_sos_decomposition(self):
        pass

    @abstractmethod
    def __getitem__(self, monomial):
        pass


class MosekSolution(BaseSolution):
    def __init__(self, relaxation, model, primal):
        self._relaxation = relaxation
        self._model = model
        self._primal = primal

    def __getitem__(self, monomial):
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

    def get_sos_decomposition(self):
        if self._primal:
            raise NotImplementedError("Can't derive the SOS decomposition by solving the primal for now")
