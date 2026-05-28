from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from ncpoleon.polynomials import _Polynomial, _PolynomialElements, _Scalar
from ncpoleon.relaxations import _BaseSdpRelaxation

if TYPE_CHECKING:
    import picos as pc
    from mosek.fusion import Model

class BaseSolution(Generic[_PolynomialElements, _Scalar]):
    def get_sos_decomposition(self) -> _Polynomial[_PolynomialElements, _Scalar]: ...
    def __getitem__(self, monomial: _PolynomialElements) -> _Scalar: ...

class MosekSolution(BaseSolution[_PolynomialElements, _Scalar]):
    def __init__(self, relaxation: _BaseSdpRelaxation[_PolynomialElements, _Scalar], model: Model, primal: bool): ...

class PicosSolution(BaseSolution[_PolynomialElements, _Scalar]):
    def __init__(
        self, relaxation: _BaseSdpRelaxation[_PolynomialElements, _Scalar], problem: pc.modeling.Problem, primal: bool
    ): ...
