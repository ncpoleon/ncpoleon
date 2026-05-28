use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;

use crate::polynomials::monomial::OneWithMomentMatrixId;
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::{
    PythonNonCommutativeMonomial, RustNonCommutativeMonomial,
};
use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::{
    PythonNonCommutativeOperator, RustNonCommutativeOperator,
};

/// Generate a list of non-commutative operator variables.
///
/// Returns `number` [`NonCommutativeOperator`] instances.
///
/// # Arguments
/// * `label` – Single character used as the variable name prefix.
/// * `number` – How many variables to generate.
/// * `moment_matrix_id` – The index of the moment matrix the variables are associated to.
/// * `starting_index` – Index assigned to the first variable (default: `0`).
/// * `hermitian` – If `True`, the operators are marked as Hermitian (self-adjoint).
/// * `projector` – If `True`, the operators are marked as projectors. Setting this also implies `hermitian=True`.
/// * `return_identity` – If `True`, returns a `(operators, identity)` tuple instead of just the list. The identity
///   operator for that moment matrix index is notably useful to specify normalization constraints when using multiple
///   moment matrices.
#[pyfunction]
#[pyo3(signature=(label, number, *, moment_matrix_id=0, starting_index = 0, hermitian=false, projector=false, return_identity=false))]
pub(crate) fn generate_noncommutative_variables(
    label: char,
    number: u8,
    moment_matrix_id: u8,
    starting_index: u8,
    hermitian: bool,
    projector: bool,
    return_identity: bool,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let operators: Vec<Py<PyAny>> = (starting_index..(starting_index + number))
            .map(|index| {
                PythonNonCommutativeOperator(RustNonCommutativeOperator::new(
                    label,
                    index,
                    false,
                    hermitian | projector,
                    projector,
                    moment_matrix_id,
                ))
                .into_py_any(py)
            })
            .collect::<PyResult<_>>()?;

        if return_identity {
            let identity: Py<PyAny> =
                PythonNonCommutativeMonomial(RustNonCommutativeMonomial::one(moment_matrix_id)).into_py_any(py)?;
            (operators, identity).into_py_any(py)
        } else {
            operators.into_py_any(py)
        }
    })
}
