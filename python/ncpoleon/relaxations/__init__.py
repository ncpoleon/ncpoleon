from ncpoleon._accelerate.relaxations import (
    Canonicality,
    ComplexCoefficientsCommutativeConstraint,
    ComplexCoefficientsNonCommutativeConstraint,
    ComplexValuedCommutativeMomentMatrix,
    ComplexValuedCommutativeSdpRelaxation,
    ComplexValuedNonCommutativeMomentMatrix,
    ComplexValuedNonCommutativeSdpRelaxation,
    Hermiticity,
    RealCoefficientsCommutativeConstraint,
    RealCoefficientsNonCommutativeConstraint,
    RealValuedCommutativeMomentMatrix,
    RealValuedCommutativeSdpRelaxation,
    RealValuedNonCommutativeMomentMatrix,
    RealValuedNonCommutativeSdpRelaxation,
    Realness,
)
from ncpoleon._accelerate.relaxations import (
    get_relaxation as _get_relaxation,
)
from ncpoleon.logging import set_verbosity_level


def get_relaxation(variables, level, objective, *, verbosity=0, **kwargs):
    set_verbosity_level(verbosity)
    return _get_relaxation(variables, level, objective, verbosity=verbosity, **kwargs)


__all__ = [
    "get_relaxation",
    "Realness",
    "Canonicality",
    "Hermiticity",
    "RealCoefficientsCommutativeConstraint",
    "ComplexCoefficientsCommutativeConstraint",
    "RealCoefficientsNonCommutativeConstraint",
    "ComplexCoefficientsNonCommutativeConstraint",
    "RealValuedCommutativeMomentMatrix",
    "ComplexValuedCommutativeMomentMatrix",
    "RealValuedNonCommutativeMomentMatrix",
    "ComplexValuedNonCommutativeMomentMatrix",
    "RealValuedCommutativeSdpRelaxation",
    "ComplexValuedCommutativeSdpRelaxation",
    "RealValuedNonCommutativeSdpRelaxation",
    "ComplexValuedNonCommutativeSdpRelaxation",
]
