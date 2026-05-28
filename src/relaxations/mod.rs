pub(crate) mod constraint;
mod moment_matrix;
mod sdp_relaxation;

use pyo3::prelude::*;

#[pymodule]
pub fn relaxations(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<constraint::PythonRealCoefficientsCommutativeConstraint>()?;
    m.add_class::<constraint::PythonComplexCoefficientsCommutativeConstraint>()?;
    m.add_class::<constraint::PythonRealCoefficientsNonCommutativeConstraint>()?;
    m.add_class::<constraint::PythonComplexCoefficientsNonCommutativeConstraint>()?;
    m.add_class::<moment_matrix::PythonRealValuedCommutativeMomentMatrix>()?;
    m.add_class::<moment_matrix::PythonComplexValuedCommutativeMomentMatrix>()?;
    m.add_class::<moment_matrix::PythonRealValuedNonCommutativeMomentMatrix>()?;
    m.add_class::<moment_matrix::PythonComplexValuedNonCommutativeMomentMatrix>()?;
    m.add_class::<sdp_relaxation::PythonRealValuedCommutativeSdpRelaxation>()?;
    m.add_class::<sdp_relaxation::PythonComplexValuedCommutativeSdpRelaxation>()?;
    m.add_class::<sdp_relaxation::PythonRealValuedNonCommutativeSdpRelaxation>()?;
    m.add_class::<sdp_relaxation::PythonComplexValuedNonCommutativeSdpRelaxation>()?;
    m.add_function(wrap_pyfunction!(sdp_relaxation::get_relaxation, m)?)?;
    Ok(())
}
