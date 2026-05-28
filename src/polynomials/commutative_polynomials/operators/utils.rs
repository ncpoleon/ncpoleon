use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::{
    PythonCommutativeMonomial, RustCommutativeMonomial,
};
use crate::polynomials::commutative_polynomials::operators::commutative_operator::{
    PythonCommutativeOperator, RustCommutativeOperator,
};
use crate::polynomials::monomial::OneWithMomentMatrixId;

/// Generate a list of commutative scalar variables.
///
/// Returns `number` [`CommutativeOperator`] instances.
///
/// # Arguments
/// * `label` – Single character used as the variable name prefix.
/// * `number` – How many variables to generate.
/// * `moment_matrix_id` – The index of the moment matrix the variables are associated to.
/// * `starting_index` – Index assigned to the first variable (default: `0`).
/// * `real` – If `True`, the variables are marked as real-valued.
/// * `projector` – If `True`, the variables are projectors (`x ** 2 == x`). Setting this also implies `real=True`.
/// * `return_identity` – If `True`, returns a `(operators, identity)` tuple instead of just the list. The identity
///   operator for that moment matrix index is notably useful to specify normalization constraints when using multiple
///   moment matrices.
#[pyfunction]
#[pyo3(signature=(label, number, *, moment_matrix_id=0, starting_index = 0, real=false, projector=false, return_identity=false))]
pub(crate) fn generate_commutative_variables(
    label: char,
    number: u8,
    moment_matrix_id: u8,
    starting_index: u8,
    real: bool,
    projector: bool,
    return_identity: bool,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let operators: Vec<Py<PyAny>> = (starting_index..(starting_index + number))
            .map(|index| {
                PythonCommutativeOperator(RustCommutativeOperator::new(
                    label,
                    index,
                    false,
                    real | projector,
                    projector,
                    moment_matrix_id,
                ))
                .into_py_any(py)
            })
            .collect::<PyResult<_>>()?;

        if return_identity {
            let identity: Py<PyAny> =
                PythonCommutativeMonomial(RustCommutativeMonomial::one(moment_matrix_id)).into_py_any(py)?;
            (operators, identity).into_py_any(py)
        } else {
            operators.into_py_any(py)
        }
    })
}
