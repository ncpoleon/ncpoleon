use log::LevelFilter;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use pyo3_log::{Caching, Logger};

mod logging;
mod polynomials;
mod progress;
mod relaxations;

/// A Python module implemented in Rust.
#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    if logging::LOGGER.get().is_none() {
        logging::LOGGER
            .set(
                Logger::new(m.py(), Caching::LoggersAndLevels)?
                    .filter(LevelFilter::Trace)
                    .install()
                    .map_err(|_| PyRuntimeError::new_err("Failed to install logger."))?,
            )
            .map_err(|_| PyRuntimeError::new_err("Failed to set logger."))?;
    }

    // Inserting to sys.modules allows importing submodules nicely from Python
    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;

    m.add_wrapped(wrap_pymodule!(logging::logging))?;
    sys_modules.set_item("ncpoleon._accelerate.logging", m.getattr("logging")?)?;

    m.add_wrapped(wrap_pymodule!(polynomials::polynomials))?;
    sys_modules.set_item("ncpoleon._accelerate.polynomials", m.getattr("polynomials")?)?;

    m.add_wrapped(wrap_pymodule!(relaxations::relaxations))?;
    sys_modules.set_item("ncpoleon._accelerate.relaxations", m.getattr("relaxations")?)?;

    Ok(())
}
