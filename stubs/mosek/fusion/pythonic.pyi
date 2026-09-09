"""Type stubs for ``mosek.fusion.pythonic``, MOSEK 11.2.

At runtime this module does two things: it re-exports everything from
``mosek.fusion`` (``from mosek.fusion import *``) and it monkey-patches the
arithmetic, comparison and indexing dunders — plus ``.T``, ``.F`` and
``.shape`` — onto ``Expression``, ``Variable``, ``Matrix`` and ``Constraint``.

A stub cannot express "importing this module adds members to that class", so
those members are declared directly on the classes in ``mosek/fusion/__init__.pyi``
and this file only mirrors the star re-export. Importing it is what makes them
real, which is why every module here that uses Fusion operators imports it.
"""

from mosek.fusion import *  # noqa: F403
