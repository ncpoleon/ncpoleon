use pyo3::prelude::*;

pub(crate) mod commutative_monomial;

#[pymodule]
pub fn _commutative_monomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<commutative_monomial::PythonCommutativeMonomial>()?;
    Ok(())
}
