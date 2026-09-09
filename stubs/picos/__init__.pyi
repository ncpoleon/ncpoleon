"""Type stubs for ``picos``, PICOS 2.6.2.

PICOS ships no ``py.typed``, so a type checker sees nothing of it. Everything
here was recovered from the installed package -- ``dir(picos)`` for the export
list, ``__mro__`` for the hierarchy, ``inspect.signature`` for the signatures,
and a real solve in a shell for the return types -- and the layout mirrors
PICOS's own: the definitions live in ``picos.expressions``,
``picos.constraints``, ``picos.modeling`` and ``picos.valuable``, and this
module re-exports them the way ``picos/__init__.py`` does.

Note that PICOS does *not* re-export ``Expression`` or ``Constraint`` at the
top level, so there is deliberately no ``picos.Expression`` /
``picos.Constraint`` here either. Annotate with the real paths --
``pc.expressions.Expression`` and ``pc.constraints.Constraint`` -- which this
module re-exports as submodules, the same way ``pc.modeling.Problem`` works.
"""

from picos import constraints as constraints
from picos import expressions as expressions
from picos import modeling as modeling
from picos import valuable as valuable

from picos.constraints import FlowConstraint as FlowConstraint
from picos.expressions import (
    AffineExpression as AffineExpression,
    Ball as Ball,
    BaseVariable as BaseVariable,
    BiaffineExpression as BiaffineExpression,
    BinaryVariable as BinaryVariable,
    ComplexAffineExpression as ComplexAffineExpression,
    ComplexVariable as ComplexVariable,
    Constant as Constant,
    DetRootN as DetRootN,
    Ellipsoid as Ellipsoid,
    Entropy as Entropy,
    ExponentialCone as ExponentialCone,
    GeometricMean as GeometricMean,
    HermitianVariable as HermitianVariable,
    I as I,
    IntegerVariable as IntegerVariable,
    J as J,
    LogSumExp as LogSumExp,
    Logarithm as Logarithm,
    LowerTriangularVariable as LowerTriangularVariable,
    Mutable as Mutable,
    NegativeEntropy as NegativeEntropy,
    NonnegativeOrthant as NonnegativeOrthant,
    Norm as Norm,
    NuclearNorm as NuclearNorm,
    O as O,
    PositiveSemidefiniteCone as PositiveSemidefiniteCone,
    PowerTrace as PowerTrace,
    ProductCone as ProductCone,
    RealVariable as RealVariable,
    RotatedSecondOrderCone as RotatedSecondOrderCone,
    Samples as Samples,
    SecondOrderCone as SecondOrderCone,
    Simplex as Simplex,
    SkewSymmetricVariable as SkewSymmetricVariable,
    SpectralNorm as SpectralNorm,
    SquaredNorm as SquaredNorm,
    SumExponentials as SumExponentials,
    SumExtremes as SumExtremes,
    SymmetricVariable as SymmetricVariable,
    TheField as TheField,
    UpperTriangularVariable as UpperTriangularVariable,
    ZeroSpace as ZeroSpace,
    block as block,
    diag as diag,
    diag_vect as diag_vect,
    exp as exp,
    kron as kron,
    log as log,
    maindiag as maindiag,
    max as max,
    min as min,
    new_param as new_param,
    norm as norm,
    partial_trace as partial_trace,
    partial_transpose as partial_transpose,
    sum as sum,
    trace as trace,
)
from picos.modeling import (
    Objective as Objective,
    Options as Options,
    Problem as Problem,
    Solution as Solution,
    SolutionFailure as SolutionFailure,
    find_assignment as find_assignment,
    maximize as maximize,
    minimize as minimize,
)
from picos.valuable import NotValued as NotValued

__version__: str

def available_solvers(problem: object = ...) -> list[str]: ...
def patch_scipy_array_priority() -> None: ...
