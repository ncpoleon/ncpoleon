from ncpoleon._accelerate.polynomials import RewritingStrategy

from .commutative_polynomials import generate_commutative_variables
from .noncommutative_polynomials import generate_noncommutative_variables

__all__ = ["generate_commutative_variables", "generate_noncommutative_variables", "RewritingStrategy"]
