use pyo3::prelude::*;

pub(crate) mod noncommutative_polynomial;

#[pymodule]
pub fn _noncommutative_polynomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<noncommutative_polynomial::PythonRealCoefficientsNonCommutativePolynomial>()?;
    m.add_class::<noncommutative_polynomial::PythonComplexCoefficientsNonCommutativePolynomial>()?;
    Ok(())
}
