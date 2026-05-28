use pyo3::prelude::*;

pub(crate) mod noncommutative_operator;
mod utils;

#[pymodule]
pub fn _noncommutative_operators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<noncommutative_operator::PythonNonCommutativeOperator>()?;
    m.add_function(wrap_pyfunction!(utils::generate_noncommutative_variables, m)?)?;
    Ok(())
}
