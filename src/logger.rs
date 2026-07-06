use std::sync::OnceLock;

use log::debug;
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
