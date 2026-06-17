import pytest
from ncpoleon import generate_noncommutative_variables, get_relaxation
from ncpoleon.export import to_mosek, to_picos
from ncpoleon.utils import is_mosek_available

# TODO: Add complex-valued tests, tests for the attributes of the relaxations such that the equality constraints or the
# monomial index


def generate_i3322_parameters():
    for export in ["mosek", "picos"]:
        for use_primal in [True, False]:
            marks = []

            if export == "mosek":
                marks.append(
                    pytest.mark.skipif(
                        not is_mosek_available(), reason="Mosek is not installed or a Mosek license is not available."
                    )
                )

                if use_primal:
                    marks.append(
                        pytest.mark.xfail(
                            reason="Solving the primal using the MOSEK Python Fusion API may result in a Recursion "
                            "Error because the involved LMI is too large.",
                            raises=RecursionError,
                        )
                    )

            yield pytest.param(export, use_primal, marks=marks)


@pytest.mark.parametrize("export, use_primal", generate_i3322_parameters())
def test_i3322(export, use_primal: bool):
    """
    Maximize the Bell-inequality I3322.

    If p(0,0|x,y) = Tr[rho (A_{0|x} otimes B_{0|y})] is denoted pxy then the Bell inequality is

    -p00-p11-p10-p01-p02-p20+p12+p21 + p_A(0|0) + p_B(0|0)

    The quantum bound is roughly 1.25087
    """
    m0, m1, m2 = generate_noncommutative_variables("M", 3, projector=True)
    n0, n1, n2 = generate_noncommutative_variables("N", 3, projector=True)

    substitutions = {op1 * op2: op2 * op1 for op1 in [m0, m1, m2] for op2 in [n0, n1, n2]}
    obj = -m0 * n0 - m1 * n1 - m0 * n1 - m1 * n0 - m0 * n2 - m2 * n0 + m1 * n2 + m2 * n1 + m0 + n0

    sdp = get_relaxation([m0, m1, m2, n0, n1, n2], 3, obj, substitutions=substitutions)

    if export == "picos":
        problem = to_picos(sdp, "max", primal=use_primal)
    elif export == "mosek":
        problem = to_mosek(sdp, "max", primal=use_primal)
    else:
        raise ValueError(f"Unknown export: {export}.")

    problem.solve()

    if export == "picos":
        assert problem.value == pytest.approx(1.2508756)
    elif export == "mosek":
        assert problem.primalObjValue() == pytest.approx(1.2508756)
