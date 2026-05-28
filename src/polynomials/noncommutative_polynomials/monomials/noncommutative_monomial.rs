use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hasher;
use std::iter::once;
use std::ops::Mul;

use itertools::Itertools;
use num_complex::Complex;
use num_traits::Pow;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyNone;
use rustc_hash::FxHasher;

use crate::polynomials::monomial::{
    AdjointTrait, HasAMomentMatrixId, Monomial, OneWithMomentMatrixId, RewritingStrategy, RewritingTrait,
};
use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::{
    PythonNonCommutativeOperator, RustNonCommutativeOperator,
};
use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::{
    PythonComplexCoefficientsNonCommutativePolynomial, PythonRealCoefficientsNonCommutativePolynomial,
    RustNonCommutativePolynomial,
};
use crate::polynomials::polynomial::PolynomialDtype;
use crate::polynomials::utils::add::manage_entry;
use crate::relaxations::constraint::{ConstraintKind, make_noncommutative_constraint};

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct NonCommutativeMonomialDataWithMomentMatrixIndex {
    pub(crate) inner_data: Vec<RustNonCommutativeOperator>,
    pub(crate) moment_matrix_id: u8,
}

impl HasAMomentMatrixId for NonCommutativeMonomialDataWithMomentMatrixIndex {
    fn moment_matrix_id(&self) -> u8 {
        self.moment_matrix_id
    }
}

pub(crate) type RustNonCommutativeMonomial = Monomial<NonCommutativeMonomialDataWithMomentMatrixIndex>;

/// A monomial built from non-commutative operators.
///
/// A `NonCommutativeMonomial` represents an ordered product of
/// [`NonCommutativeOperator`] instances, e.g. `A(0) * A(1) * A(0)`.
#[pyclass(frozen, module = "ncpoleon.polynomials.noncommutative_polynomials", name = "NonCommutativeMonomial")]
#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct PythonNonCommutativeMonomial(pub(crate) RustNonCommutativeMonomial);

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonNonCommutativeMonomial {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(mon) = value.cast::<PythonNonCommutativeMonomial>() {
            Ok(PythonNonCommutativeMonomial(mon.get().0.clone()))
        } else if let Ok(op) = value.cast::<PythonNonCommutativeOperator>() {
            Ok(PythonNonCommutativeMonomial(op.get().0.into()))
        } else if value.extract::<f64>().is_ok_and(|f| f == 1.0) {
            // Caution! We can't know the moment matrix index when converting here, so we instead set it to zero,
            // and it is then the responsibility of the caller to sanitize the moment_matrix_id
            Ok(PythonNonCommutativeMonomial(RustNonCommutativeMonomial::one(0)))
        } else {
            Err(PyTypeError::new_err("Couldn't convert to NonCommutativeMonomial"))
        }
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for PythonNonCommutativeMonomial {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl From<PythonNonCommutativeOperator> for PythonNonCommutativeMonomial {
    fn from(value: PythonNonCommutativeOperator) -> Self {
        Self(value.0.into())
    }
}

#[pymethods]
impl PythonNonCommutativeMonomial {
    // TODO: could be considerably simplified by using the TryInto trait from Bound to polynomial
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 + &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 + &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 + &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 + &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial((&self.0 + lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(
                (&self.0 + lambda_complex).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial((&self.0 + lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(
                (&self.0 + lambda_complex).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __sub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 - &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 - &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 - &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 - &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial((&self.0 - lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(
                (&self.0 - lambda_complex).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rsub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial((-&self.0 + lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(
                (-&self.0 + lambda_complex).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __mul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            PythonNonCommutativeMonomial((&self.0 * op.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            PythonNonCommutativeMonomial((&self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            PythonRealCoefficientsNonCommutativePolynomial(
                (&self.0 * &poly_real.get().0).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            PythonComplexCoefficientsNonCommutativePolynomial(
                (&self.0 * &poly_complex.get().0).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsNonCommutativePolynomial(&self.0 * (1.0 / lambda_real)).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 * (1.0 / lambda_complex)).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    pub(crate) fn __str__(&self) -> String {
        self.0.__str__()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __neg__(&self) -> PythonRealCoefficientsNonCommutativePolynomial {
        PythonRealCoefficientsNonCommutativePolynomial(-&self.0)
    }

    fn __pow__<'py>(&self, power: u8, _modulo: &Bound<'py, PyNone>) -> PyResult<PythonNonCommutativeMonomial> {
        Ok(Self((&self.0).pow(power).map_err(PyValueError::new_err)?))
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = self.clone().into_py_any(py)?;
        make_noncommutative_constraint(self_any.bind(py), other, ConstraintKind::Equality)
    }

    fn __ge__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = self.clone().into_py_any(py)?;
        make_noncommutative_constraint(self_any.bind(py), other, ConstraintKind::Inequality)
    }

    fn __le__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = self.clone().into_py_any(py)?;
        make_noncommutative_constraint(other, self_any.bind(py), ConstraintKind::Inequality)
    }

    fn __hash__(&self) -> u64 {
        self.0.__hash__()
    }

    fn __len__(&self) -> usize {
        self.0.len() as usize
    }

    #[getter]
    fn moment_matrix_id(&self) -> u8 {
        self.0.data.moment_matrix_id
    }

    /// Return the adjoint (Hermitian conjugate) of this monomial.
    fn adjoint(&self) -> PythonNonCommutativeMonomial {
        Self(self.0.adjoint())
    }
}

impl RustNonCommutativeMonomial {
    pub fn new(data: Vec<RustNonCommutativeOperator>, moment_matrix_id: u8) -> Self {
        Self { data: NonCommutativeMonomialDataWithMomentMatrixIndex { inner_data: data, moment_matrix_id } }
    }

    pub fn len(&self) -> u8 {
        self.data.inner_data.len() as u8
    }

    pub(crate) fn __str__(&self) -> String {
        self.to_string()
    }

    pub(crate) fn __hash__(&self) -> u64 {
        if self.data.inner_data.len() == 1 {
            let operator = &self.data.inner_data[0];
            let mut hasher = FxHasher::default();
            hasher.write_u8(operator.id.index);
            hasher.write_u8(operator.id.label as u8);
            hasher.write_u8(operator.id.is_adjoint as u8);
            hasher.write_u8(operator.id.moment_matrix_id);
            return hasher.finish();
        }

        let mut hasher = FxHasher::default();
        for operator in self.data.inner_data.iter() {
            hasher.write_u8(operator.id.index);
            hasher.write_u8(operator.id.label as u8);
            hasher.write_u8(operator.id.is_adjoint as u8);
            hasher.write_u8(operator.id.is_hermitian as u8);
            hasher.write_u8(operator.id.is_projector as u8);
        }
        hasher.write_u8(self.moment_matrix_id());
        hasher.finish()
    }

    /// Check whether a noncommutative monomial can be reduced under a given substitution rule
    /// by finding the substitution key as a contiguous subsequence of operators.
    fn can_be_reduced(&self, substitution_rule: &(Self, Self)) -> Result<Option<Self>, String> {
        let divisor = &substitution_rule.0.data.inner_data;
        let replacement_data = &substitution_rule.1.data.inner_data;
        let mmi = self.data.moment_matrix_id;

        if divisor.len() > self.data.inner_data.len() {
            return Ok(None);
        }

        for start in 0..=(self.data.inner_data.len() - divisor.len()) {
            if self.data.inner_data[start..start + divisor.len()] == *divisor {
                let prefix = Self::new(self.data.inner_data[..start].to_vec(), mmi);
                let replacement = Self::new(replacement_data.to_vec(), substitution_rule.1.data.moment_matrix_id);
                let suffix = Self::new(self.data.inner_data[start + divisor.len()..].to_vec(), mmi);
                let tmp = (prefix * &replacement)?;
                return Ok(Some((tmp * &suffix)?));
            }
        }

        Ok(None)
    }
}

impl fmt::Display for RustNonCommutativeMonomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.inner_data.is_empty() {
            return write!(f, "I_{}", self.moment_matrix_id());
        }
        for operator in &self.data.inner_data {
            write!(f, "{}", operator)?;
        }
        Ok(())
    }
}

impl AdjointTrait for RustNonCommutativeMonomial {
    fn adjoint(&self) -> Self {
        Self::new(
            self.data.inner_data.iter().cloned().rev().map(|operator| operator.adjoint()).collect(),
            self.data.moment_matrix_id,
        )
    }
}

impl OneWithMomentMatrixId for RustNonCommutativeMonomial {
    fn one(moment_matrix_id: u8) -> Self {
        Self::new(Vec::new(), moment_matrix_id)
    }

    fn is_one(&self) -> bool {
        self.data.inner_data.is_empty()
    }

    fn identity_symbol() -> Option<&'static str> {
        Some("I")
    }
}

impl RewritingTrait<Self> for RustNonCommutativeMonomial {
    fn rewrite(&self, strategy: RewritingStrategy, substitutions: &BTreeMap<Self, Self>) -> Result<Self, String> {
        match strategy {
            RewritingStrategy::None => Ok(self.clone()),
            RewritingStrategy::Greedy => {
                let sorted_substitutions: Vec<(Self, Self)> = substitutions
                    .iter()
                    .map(|(mon1, mon2)| (mon1.clone(), mon2.clone()))
                    .sorted_by_key(|mon| mon.1.len() as i16 - mon.0.len() as i16)
                    .collect();
                let mut current = self.clone();

                'outer: loop {
                    for substitution_rule in sorted_substitutions.iter() {
                        if let Some(res) = current.can_be_reduced(substitution_rule)? {
                            current = res;
                            continue 'outer;
                        }
                    }
                    break;
                }

                Ok(current)
            }
        }
    }
}

impl From<&RustNonCommutativeOperator> for RustNonCommutativeMonomial {
    fn from(item: &RustNonCommutativeOperator) -> Self {
        Self::new(vec![*item], item.id.moment_matrix_id)
    }
}

impl From<RustNonCommutativeOperator> for RustNonCommutativeMonomial {
    fn from(item: RustNonCommutativeOperator) -> Self {
        Self::new(vec![item], item.id.moment_matrix_id)
    }
}

impl Mul<&RustNonCommutativeOperator> for &RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(self, rhs: &RustNonCommutativeOperator) -> Self::Output {
        let last = self.data.inner_data.last();
        match last {
            None => Ok(RustNonCommutativeMonomial::from(rhs)),
            Some(op) => {
                if op.id.moment_matrix_id != rhs.id.moment_matrix_id {
                    return Err(format!(
                        "Cannot multiply operators from different parties: {} and {}",
                        op.id.moment_matrix_id, rhs.id.moment_matrix_id
                    ));
                }
                Ok(if (rhs == op) & rhs.id.is_projector {
                    self.clone()
                } else {
                    let mut res = Vec::with_capacity(1 + self.data.inner_data.len());
                    res.extend(self.data.inner_data.to_vec());
                    res.push(*rhs);
                    RustNonCommutativeMonomial::new(res, self.data.moment_matrix_id)
                })
            }
        }
    }
}

#[allow(clippy::op_ref)]
impl Mul<RustNonCommutativeOperator> for &RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(self, rhs: RustNonCommutativeOperator) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&RustNonCommutativeOperator> for RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(mut self, rhs: &RustNonCommutativeOperator) -> Self::Output {
        let last = self.data.inner_data.last().copied();
        match last {
            None => Ok(RustNonCommutativeMonomial::from(rhs)),
            Some(op) => {
                if op.id.moment_matrix_id != rhs.id.moment_matrix_id {
                    return Err(format!(
                        "Cannot multiply operators from different parties: {} and {}",
                        op.id.moment_matrix_id, rhs.id.moment_matrix_id
                    ));
                }
                if (rhs != &op) | !rhs.id.is_projector {
                    self.data.inner_data.push(*rhs);
                }
                Ok(self)
            }
        }
    }
}

impl Mul<&RustNonCommutativeMonomial> for &RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(self, rhs: &RustNonCommutativeMonomial) -> Self::Output {
        let last_self = self.data.inner_data.last();
        match last_self {
            None => {
                if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                    return Err(format!(
                        "Cannot multiply monomials from different moment matrix indices: {} and {}",
                        self.data.moment_matrix_id, rhs.data.moment_matrix_id
                    ));
                }
                Ok(rhs.clone())
            }
            Some(op_self) => {
                let mut rhs_iter = rhs.data.inner_data.iter().cloned();
                let first_rhs = rhs_iter.next();
                match first_rhs {
                    None => {
                        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different moment matrix indices: {} and {}",
                                self.data.moment_matrix_id, rhs.data.moment_matrix_id
                            ));
                        }
                        Ok(self.clone())
                    }
                    Some(op_rhs) => {
                        if op_self.id.moment_matrix_id != op_rhs.id.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different parties: {} and {}",
                                op_self.id.moment_matrix_id, op_rhs.id.moment_matrix_id
                            ));
                        }
                        Ok(if (op_self == &op_rhs) & op_self.id.is_projector {
                            RustNonCommutativeMonomial::new(
                                self.data.inner_data.iter().cloned().chain(rhs_iter).collect(),
                                self.data.moment_matrix_id,
                            )
                        } else {
                            RustNonCommutativeMonomial::new(
                                self.data.inner_data.iter().cloned().chain(once(op_rhs)).chain(rhs_iter).collect(),
                                self.data.moment_matrix_id,
                            )
                        })
                    }
                }
            }
        }
    }
}

impl Mul<RustNonCommutativeMonomial> for &RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(self, mut rhs: RustNonCommutativeMonomial) -> Self::Output {
        let last_self = self.data.inner_data.last();

        match last_self {
            None => {
                if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                    return Err(format!(
                        "Cannot multiply monomials from different moment matrix indices: {} and {}",
                        self.data.moment_matrix_id, rhs.data.moment_matrix_id
                    ));
                }
                Ok(rhs)
            }
            Some(op_self) => {
                let first_rhs = rhs.data.inner_data.first().copied();

                match first_rhs {
                    None => {
                        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different moment matrix indices: {} and {}",
                                self.data.moment_matrix_id, rhs.data.moment_matrix_id
                            ));
                        }
                        Ok(self.clone())
                    }
                    Some(op_rhs) => {
                        if op_self.id.moment_matrix_id != op_rhs.id.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different parties: {} and {}",
                                op_self.id.moment_matrix_id, op_rhs.id.moment_matrix_id
                            ));
                        }
                        if (op_self == &op_rhs) & op_self.id.is_projector {
                            rhs.data.inner_data.splice(
                                0..0,
                                self.data.inner_data.iter().cloned().take(self.data.inner_data.len() - 1),
                            );
                        } else {
                            rhs.data.inner_data.splice(0..0, self.data.inner_data.to_vec());
                        }
                        rhs.data.moment_matrix_id = self.data.moment_matrix_id;
                        Ok(rhs)
                    }
                }
            }
        }
    }
}

impl Mul<&RustNonCommutativeMonomial> for RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(mut self, rhs: &RustNonCommutativeMonomial) -> Self::Output {
        let last_self = self.data.inner_data.last().copied();

        match last_self {
            None => {
                if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                    return Err(format!(
                        "Cannot multiply monomials from different moment matrix indices: {} and {}",
                        self.data.moment_matrix_id, rhs.data.moment_matrix_id
                    ));
                }
                Ok(rhs.clone())
            }
            Some(op_self) => {
                let first_rhs = rhs.data.inner_data.first();

                match first_rhs {
                    None => {
                        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different moment matrix indices: {} and {}",
                                self.data.moment_matrix_id, rhs.data.moment_matrix_id
                            ));
                        }
                        Ok(self)
                    }
                    Some(op_rhs) => {
                        if op_self.id.moment_matrix_id != op_rhs.id.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different parties: {} and {}",
                                op_self.id.moment_matrix_id, op_rhs.id.moment_matrix_id
                            ));
                        }
                        let len = self.data.inner_data.len();
                        if (op_self == *op_rhs) & op_self.id.is_projector {
                            self.data.inner_data.splice(len..len, rhs.data.inner_data.iter().cloned().skip(1));
                        } else {
                            self.data.inner_data.splice(len..len, rhs.data.inner_data.to_vec());
                        }
                        Ok(self)
                    }
                }
            }
        }
    }
}

impl Mul<RustNonCommutativeMonomial> for RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn mul(self, mut rhs: RustNonCommutativeMonomial) -> Self::Output {
        let last_self = self.data.inner_data.last().copied();

        match last_self {
            None => {
                if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                    return Err(format!(
                        "Cannot multiply monomials from different moment matrix indices: {} and {}",
                        self.data.moment_matrix_id, rhs.data.moment_matrix_id
                    ));
                }
                Ok(rhs)
            }
            Some(op_self) => {
                let first_rhs = rhs.data.inner_data.first().copied();

                match first_rhs {
                    None => {
                        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different moment matrix indices: {} and {}",
                                self.data.moment_matrix_id, rhs.data.moment_matrix_id
                            ));
                        }
                        Ok(self)
                    }
                    Some(op_rhs) => {
                        if op_self.id.moment_matrix_id != op_rhs.id.moment_matrix_id {
                            return Err(format!(
                                "Cannot multiply monomials from different parties: {} and {}",
                                op_self.id.moment_matrix_id, op_rhs.id.moment_matrix_id
                            ));
                        }
                        if (op_self == op_rhs) & op_self.id.is_projector {
                            rhs.data.inner_data.splice(
                                0..0,
                                self.data.inner_data.iter().cloned().take(self.data.inner_data.len() - 1),
                            );
                        } else {
                            rhs.data.inner_data.splice(0..0, self.data.inner_data.to_vec());
                        }
                        rhs.data.moment_matrix_id = self.data.moment_matrix_id;
                        Ok(rhs)
                    }
                }
            }
        }
    }
}

impl<Scalar: PolynomialDtype> Mul<&RustNonCommutativePolynomial<Scalar>> for &RustNonCommutativeMonomial {
    type Output = Result<RustNonCommutativePolynomial<Scalar>, String>;

    fn mul(self, rhs: &RustNonCommutativePolynomial<Scalar>) -> Self::Output {
        let mut res = BTreeMap::new();

        for (mon, &coeff) in rhs.data.iter() {
            manage_entry(&mut res, (self * mon)?, coeff);
        }

        Ok(RustNonCommutativePolynomial { data: res })
    }
}

#[allow(clippy::op_ref)]
#[cfg(test)]
mod tests {
    use num_complex::Complex;
    use num_traits::Zero;
    use rstest::{fixture, rstest};

    use super::*;
    use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::RustNonCommutativeOperator;
    use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::RustComplexCoefficientsNonCommutativePolynomial;

    #[fixture]
    fn mon() -> RustNonCommutativeMonomial {
        RustNonCommutativeMonomial::new(
            vec![
                RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
            ],
            0,
        )
    }

    #[rstest]
    fn test_adjoint(mon: RustNonCommutativeMonomial) {
        let expected = RustNonCommutativeMonomial::new(
            vec![
                RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
                RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                RustNonCommutativeOperator::new('x', 0, true, false, false, 0),
            ],
            0,
        );
        assert_eq!(mon.adjoint(), expected);
        assert_eq!(mon, expected.adjoint());
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsNonCommutativePolynomial::zero())]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([(
                mon.clone(),
                rhs,
            )]),
        }
    )]
    fn test_mul_complex(
        mon: RustNonCommutativeMonomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&mon * rhs, expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
        ], 0)
    )]
    #[case(
        RustNonCommutativeOperator::new('x', 3, false, true, true, 0),
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
            RustNonCommutativeOperator::new('x', 3, false, true, true, 0),
        ], 0)
    )]
    fn test_mul_operator(
        mon: RustNonCommutativeMonomial,
        #[case] rhs: RustNonCommutativeOperator,
        #[case] expected: RustNonCommutativeMonomial,
    ) {
        assert_eq!((&mon * &rhs).unwrap(), expected);
        assert_eq!((&mon * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeMonomial::one(0),
        mon.clone(),
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
        ], 0),
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
        ], 0)
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 3, false, true, true, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
        ], 0),
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
            RustNonCommutativeOperator::new('x', 3, false, true, true, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
            RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
        ], 0)
    )]
    fn test_mul_monomial(
        mon: RustNonCommutativeMonomial,
        #[case] rhs: RustNonCommutativeMonomial,
        #[case] expected: RustNonCommutativeMonomial,
    ) {
        assert_eq!((&mon * &rhs).unwrap(), expected);
        assert_eq!((&mon * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial::zero(),
        RustComplexCoefficientsNonCommutativePolynomial::zero()
    )]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new(
                            'x',
                            2,
                            false,
                            true,
                            true,
                            0
                        ),
                    ], 0),
                    Complex { re: -2.5, im: 2.5 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new(
                            'x',
                            3,
                            false,
                            true,
                            true,
                            0
                        ),
                    ], 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new(
                            'x',
                            0,
                            false,
                            false,
                            false,
                            0
                        ),
                        RustNonCommutativeOperator::new(
                            'x',
                            1,
                            false,
                            true,
                            false,
                            0
                        ),
                        RustNonCommutativeOperator::new(
                            'x',
                            2,
                            false,
                            true,
                            true,
                            0
                        ),
                    ], 0),
                    Complex { re: 1.0, im: 1.0 }
                ),
                (
                    RustNonCommutativeMonomial::one(0),
                    Complex { re: 1.5, im: -3.5 },
                ),
            ]),
        },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new(
                            'x',
                            0,
                            false,
                            false,
                            false,
                            0
                        ),
                        RustNonCommutativeOperator::new(
                            'x',
                            1,
                            false,
                            true,
                            false,
                            0
                        ),
                        RustNonCommutativeOperator::new(
                            'x',
                            2,
                            false,
                            true,
                            true,
                            0
                        ),
                        RustNonCommutativeOperator::new(
                            'x',
                            3,
                            false,
                            true,
                            true,
                            0
                        ),
                    ], 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
                    ], 0),
                    Complex { re: 1.0, im: 1.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        RustNonCommutativeOperator::new('x', 2, false, true, true, 0),
                    ], 0),
                    Complex { re: -1.0, im: -1.0 },
                ),
            ]),
        }
    )]
    fn test_mul_polynomial(
        mon: RustNonCommutativeMonomial,
        #[case] rhs: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&mon * &rhs).unwrap(), expected);
    }
}
