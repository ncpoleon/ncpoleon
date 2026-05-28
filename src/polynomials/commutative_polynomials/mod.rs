pub(crate) mod monomials;
pub(crate) mod operators;
pub(crate) mod polynomials;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;

#[pymodule]
pub fn commutative_polynomials(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;

    // Operators module
    m.add_wrapped(wrap_pymodule!(operators::_commutative_operators))?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.commutative_polynomials._operators",
        m.getattr("_commutative_operators")?,
    )?;

    m.add_wrapped(wrap_pymodule!(monomials::_commutative_monomials))?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.commutative_polynomials._monomials",
        m.getattr("_commutative_monomials")?,
    )?;

    m.add_wrapped(wrap_pymodule!(polynomials::_commutative_polynomials))?;
    sys_modules.set_item(
        "ncpoleon._accelerate.polynomials.commutative_polynomials._polynomials",
        m.getattr("_commutative_polynomials")?,
    )?;

    Ok(())
}
