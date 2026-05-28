use pyo3::prelude::*;

pub(crate) mod commutative_polynomial;

#[pymodule]
pub fn _commutative_polynomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<commutative_polynomial::PythonRealCoefficientsCommutativePolynomial>()?;
    m.add_class::<commutative_polynomial::PythonComplexCoefficientsCommutativePolynomial>()?;
    Ok(())
}
