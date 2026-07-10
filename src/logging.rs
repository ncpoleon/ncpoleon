use std::sync::OnceLock;

use log::{debug, LevelFilter};
use pyo3::prelude::*;
use pyo3_log::ResetHandle;

pub(super) static LOGGER: OnceLock<ResetHandle> = OnceLock::new();

#[pyfunction]
pub(crate) fn reset_handler() -> PyResult<()> {
    LOGGER
        .get()
        .ok_or_else(|| pyo3::PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Logger not initialized"))?
        .reset();
    debug!("Logger has been reset.");
    Ok(())
}

// Drives the `log` crate's global max-level gate so that, at low verbosity, hot-loop `trace!`/
// `debug!` calls short-circuit *before* ever reaching pyo3-log (which would otherwise pay a cache
// lookup per call). The logger is installed with `filter(Trace)` so pyo3-log's own top filter lets
// every level through; the effective filtering is delegated here and to the Python logger level.
#[pyfunction]
fn set_max_level(verbosity: u8) {
    let filter = match verbosity {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };
    log::set_max_level(filter);
}

// FIXME: This doesn't work well, to investigate
#[pyfunction]
fn set_notebook(running: bool) -> PyResult<()> {
    kdam::set_notebook(running);
    Ok(())
}

#[pymodule]
pub fn logging(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reset_handler, m)?)?;
    m.add_function(wrap_pyfunction!(set_max_level, m)?)?;
    m.add_function(wrap_pyfunction!(set_notebook, m)?)?;
    Ok(())
}
