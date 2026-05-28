use pyo3::prelude::*;

pub(crate) mod noncommutative_monomial;

#[pymodule]
pub fn _noncommutative_monomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<noncommutative_monomial::PythonNonCommutativeMonomial>()?;
    Ok(())
}
