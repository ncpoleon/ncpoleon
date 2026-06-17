pub(crate) mod commutative_polynomials;
pub(crate) mod monomial;
pub(crate) mod noncommutative_polynomials;
pub(crate) mod operator;
pub(crate) mod polynomial;
mod utils;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

#[pymodule]
pub fn polynomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<monomial::RewritingStrategy>()?;
    m.add_wrapped(wrap_pymodule!(commutative_polynomials::commutative_polynomials))?;

    // Inserting to sys.modules allows importing submodules nicely from Python
    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;
    sys_modules
        .set_item("ncpoleon._accelerate.polynomials.commutative_polynomials", m.getattr("commutative_polynomials")?)?;

    m.add_wrapped(wrap_pymodule!(noncommutative_polynomials::noncommutative_polynomials))?;

    // Inserting to sys.modules allows importing submodules nicely from Python
    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.noncommutative_polynomials",
        m.getattr("noncommutative_polynomials")?,
    )?;
    Ok(())
}
