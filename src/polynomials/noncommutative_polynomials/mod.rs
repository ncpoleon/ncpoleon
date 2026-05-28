pub(crate) mod monomials;
pub(crate) mod operators;
pub(crate) mod polynomials;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

#[pymodule]
pub fn noncommutative_polynomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;

    // Operators module
    m.add_wrapped(wrap_pymodule!(operators::_noncommutative_operators))?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.noncommutative_polynomials._operators",
        m.getattr("_noncommutative_operators")?,
    )?;

    m.add_wrapped(wrap_pymodule!(monomials::_noncommutative_monomials))?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.noncommutative_polynomials._monomials",
        m.getattr("_noncommutative_monomials")?,
    )?;

    m.add_wrapped(wrap_pymodule!(polynomials::_noncommutative_polynomials))?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.noncommutative_polynomials._polynomials",
        m.getattr("_noncommutative_polynomials")?,
    )?;

    Ok(())
}
