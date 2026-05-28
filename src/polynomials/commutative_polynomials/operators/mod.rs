use pyo3::prelude::*;

pub(crate) mod commutative_operator;
mod utils;

#[pymodule]
pub fn _commutative_operators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<commutative_operator::PythonCommutativeOperator>()?;
    m.add_function(wrap_pyfunction!(utils::generate_commutative_variables, m)?)?;
    Ok(())
}
