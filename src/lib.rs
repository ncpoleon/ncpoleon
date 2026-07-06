use log::LevelFilter;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pymodule;
use pyo3_log::{Caching, Logger};

mod logger;
mod polynomials;
mod relaxations;

/// A Python module implemented in Rust.
#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // pyo3_log::init();
    // TODO: check how to reset the logger so that the verbosity argument can change the logging level
    logger::LOGGER
        .set(
            Logger::new(m.py(), Caching::LoggersAndLevels)?
                .filter(LevelFilter::Trace)
                .install()
                .expect("Failed to instantiate logger."),
        )
        .expect("Failed to instantiate logger handle.");

    m.add_wrapped(wrap_pymodule!(polynomials::polynomials))?;

    // Inserting to sys.modules allows importing submodules nicely from Python
    let sys = PyModule::import(m.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;
    sys_modules.set_item("ncpoleon._accelerate.polynomials", m.getattr("polynomials")?)?;

    m.add_wrapped(wrap_pymodule!(relaxations::relaxations))?;
    sys_modules.set_item("ncpoleon._accelerate.relaxations", m.getattr("relaxations")?)?;

    m.add_function(wrap_pyfunction!(logger::reset_handler, m)?)?;

    Ok(())
}
