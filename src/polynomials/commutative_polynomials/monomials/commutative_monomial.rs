use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hasher;
use std::ops::Mul;

use itertools::Itertools;
use num_complex::Complex;
use num_traits::Pow;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyNone;
use rustc_hash::FxHasher;

use crate::polynomials::commutative_polynomials::operators::commutative_operator::{
    PythonCommutativeOperator, RustCommutativeOperator,
};
use crate::polynomials::commutative_polynomials::polynomials::commutative_polynomial::{
    PythonComplexCoefficientsCommutativePolynomial, PythonRealCoefficientsCommutativePolynomial,
    RustComplexCoefficientsCommutativePolynomial, RustRealCoefficientsCommutativePolynomial,
};
use crate::polynomials::monomial::{
    AdjointTrait, HasAMomentMatrixId, Monomial, OneWithMomentMatrixId, RewritingStrategy, RewritingTrait,
};
use crate::polynomials::utils::merge_btreemaps::merge_btreemaps;
use crate::relaxations::constraint::{ConstraintKind, make_commutative_constraint};

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct CommutativeMonomialDataWithMomentMatrixIndex {
    pub(crate) inner_data: BTreeMap<RustCommutativeOperator, u8>,
    pub(crate) moment_matrix_id: u8,
}

impl HasAMomentMatrixId for CommutativeMonomialDataWithMomentMatrixIndex {
    fn moment_matrix_id(&self) -> u8 {
        self.moment_matrix_id
    }
}

pub(crate) type RustCommutativeMonomial = Monomial<CommutativeMonomialDataWithMomentMatrixIndex>;

/// A monomial built from commutative scalar variables.
///
/// A `CommutativeMonomial` represents a product of [`CommutativeOperator`]
/// instances raised to non-negative integer powers, e.g. `x_(0)^2 * x_(1)`.
#[pyclass(frozen, module = "ncpoleon.polynomials.commutative_polynomials", name = "CommutativeMonomial")]
#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct PythonCommutativeMonomial(pub(crate) RustCommutativeMonomial);

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonCommutativeMonomial {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(mon) = value.cast::<PythonCommutativeMonomial>() {
            Ok(mon.get().clone())
        } else if let Ok(op) = value.cast::<PythonCommutativeOperator>() {
            Ok((*op.get()).into())
        } else if value.extract::<f64>().is_ok_and(|f| f == 1.0) {
            // Caution! We can't know the moment matrix index when converting here, so we instead set it to zero, and
            // it is then the responsability of the caller to sanitize the moment_matrix_id
            Ok(PythonCommutativeMonomial(RustCommutativeMonomial::one(0)))
        } else {
            Err(PyTypeError::new_err("Couldn't convert to CommutativeMonomial"))
        }
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for PythonCommutativeMonomial {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl From<PythonCommutativeOperator> for PythonCommutativeMonomial {
    fn from(value: PythonCommutativeOperator) -> Self {
        Self(value.0.into())
    }
}

#[pymethods]
impl PythonCommutativeMonomial {
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonCommutativeOperator>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 + &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 + &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 + &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
            PythonComplexCoefficientsCommutativePolynomial(&self.0 + &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 + lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial((&self.0 + lambda_complex).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        // __radd__ is only called if __add__ isn't supported on other, which means that it's either
        // a scalar, or not supported
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 + lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial((&self.0 + lambda_complex).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __sub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonCommutativeOperator>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 - &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 - &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 - &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
            PythonComplexCoefficientsCommutativePolynomial(&self.0 - &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 - lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial((&self.0 - lambda_complex).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rsub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        // __rsub__ is only called if __sub__ isn't supported on other, which means that it's either
        // a scalar, or not supported
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial((-&self.0 + lambda_real).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial((-&self.0 + lambda_complex).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __mul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonCommutativeOperator>() {
            PythonCommutativeMonomial((op.get().0 * &self.0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            PythonCommutativeMonomial((&self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 * &poly_real.get().0).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
            PythonComplexCoefficientsCommutativePolynomial(
                (&self.0 * &poly_complex.get().0).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        // __rmul__ is only called if __mul__ isn't supported on other, which means that it's either
        // a scalar, or not supported
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        // Can't divide by anything else than a scalar for now
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial(&self.0 * (1.0 / lambda_real)).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial(&self.0 * (1.0 / lambda_complex)).into_py_any(py)
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

    fn __neg__(&self) -> PythonRealCoefficientsCommutativePolynomial {
        PythonRealCoefficientsCommutativePolynomial(-&self.0)
    }

    fn __pow__<'py>(&self, power: u8, modulo: &Bound<'py, PyNone>) -> PyResult<Py<PyAny>> {
        Self((&self.0).pow(power).map_err(PyValueError::new_err)?).into_py_any(modulo.py())
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = self.clone().into_py_any(py)?;
        make_commutative_constraint(self_any.bind(py), other, ConstraintKind::Equality)
    }

    fn __ge__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = self.clone().into_py_any(py)?;
        make_commutative_constraint(self_any.bind(py), other, ConstraintKind::Inequality)
    }

    fn __le__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = self.clone().into_py_any(py)?;
        make_commutative_constraint(other, self_any.bind(py), ConstraintKind::Inequality)
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

    /// Return the complex conjugate of this monomial.
    fn adjoint(&self) -> PythonCommutativeMonomial {
        Self(self.0.adjoint())
    }
}

impl RustCommutativeMonomial {
    pub fn new(data: BTreeMap<RustCommutativeOperator, u8>, moment_matrix_id: u8) -> Self {
        Self { data: CommutativeMonomialDataWithMomentMatrixIndex { inner_data: data, moment_matrix_id } }
    }

    pub fn len(&self) -> u8 {
        self.data.inner_data.values().sum()
    }

    pub(crate) fn __str__(&self) -> String {
        self.to_string()
    }

    fn __hash__(&self) -> u64 {
        // This checks ensures that an operator has the same hash as its corresponding monomial
        if self.data.inner_data.len() == 1 {
            let (operator, &power) = self.data.inner_data.iter().next().unwrap();
            if power == 1 {
                return operator.__hash__();
            }
        }

        let mut hasher = FxHasher::default();

        for (operator, &operator_power) in self.data.inner_data.iter() {
            hasher.write_u64(operator.__hash__());
            hasher.write_u8(operator_power);
        }

        hasher.write_u8(self.moment_matrix_id());
        hasher.finish()
    }

    // FIXME: mostly repeated code from merge_btreemaps.rs
    /// Check whether a commutative monomial can be reduced under a given substitution rule and
    /// return the result if so
    fn can_be_reduced(&self, substitution_rule: &(Self, Self)) -> Result<Option<Self>, String> {
        if substitution_rule.0.data.moment_matrix_id != substitution_rule.1.data.moment_matrix_id {
            return Err(format!(
                "Cannot substitute monomial {} with moment matrix index {} to monomial {} with moment matrix index {}.",
                substitution_rule.0,
                substitution_rule.0.data.moment_matrix_id,
                substitution_rule.1,
                substitution_rule.1.data.moment_matrix_id
            ));
        }
        let mut res = BTreeMap::new();
        let mut self_iter = self.data.inner_data.iter().map(|(&op, &op_power)| (op, op_power));
        // Transforms into owned values
        let mut divisor_iter = substitution_rule.0.data.inner_data.iter().map(|(&op, &op_power)| (op, op_power));

        let mut self_elt = self_iter.next();
        let mut divisor_elt = divisor_iter.next();

        loop {
            let self_elt_owned = self_elt.take();
            let divisor_elt_owned = divisor_elt.take();

            // match moves the variable, so since we want to own the variable so that we don't
            // re-clone the key, we re-create a variable each time
            match (self_elt_owned, divisor_elt_owned) {
                (Some((self_op, self_op_power)), Some((divisor_op, divisor_op_power))) => {
                    let insert_op;
                    let new_power;

                    match self_op.cmp(&divisor_op) {
                        // The operator is present in the original monomial, but not in the divider,
                        // so we can keep the same power
                        Ordering::Less => {
                            new_power = self_op_power;
                            insert_op = self_op;
                            self_elt = self_iter.next();
                            divisor_elt = Some((divisor_op, divisor_op_power));
                        }
                        // The operator is present in both the original monomial and its divider, so
                        // we have to check that its associated power in the original monomial is
                        // larger than that of the divider
                        Ordering::Equal => {
                            if self_op_power < divisor_op_power {
                                return Ok(None);
                            }
                            new_power = self_op_power - divisor_op_power;
                            insert_op = self_op;
                            self_elt = self_iter.next();
                            divisor_elt = divisor_iter.next();
                        }
                        // The operator is present in the divider but not in the original monomial,
                        // so we know the divider can't divide the original monomial
                        Ordering::Greater => return Ok(None),
                    }

                    if new_power != 0 {
                        res.insert(insert_op, new_power);
                    }
                }
                // We have exhausted the divider, so we can just insert the rest of the original
                // monomial
                (Some((self_op, self_op_power)), None) => {
                    res.insert(self_op, self_op_power);
                    res.extend(self_iter);
                    break;
                }
                // We have exhausted the original monomial but not the divider, so we know the
                // latter can't divide the former
                (None, Some(_)) => return Ok(None),
                (None, None) => break,
            }
        }

        Ok(Some((RustCommutativeMonomial::new(res, substitution_rule.0.data.moment_matrix_id) * &substitution_rule.1)?))
    }
}

impl fmt::Display for RustCommutativeMonomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.inner_data.is_empty() {
            return write!(f, "1");
        }
        for (operator, &operator_power) in &self.data.inner_data {
            if operator_power == 1 {
                write!(f, "{}", operator)?;
            } else {
                write!(f, "{}^{}", operator, operator_power)?;
            }
        }
        Ok(())
    }
}

impl RewritingTrait<Self> for RustCommutativeMonomial {
    /// Rewrite a commutative monomial according to a rewriting strategy and a set of substitution
    /// rules.
    ///
    /// The first returned object indicates whether any transformation has been applied, while the
    /// second one is the modified monomial.
    fn rewrite(&self, strategy: RewritingStrategy, substitutions: &BTreeMap<Self, Self>) -> Result<Self, String> {
        match strategy {
            RewritingStrategy::None => Ok(self.clone()),
            // We apply substitution rules sorted by how much they decrease the length of the
            // monomial
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
                            // We go back to the outer loop so that we can potentially apply better
                            // substitution rules
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

impl AdjointTrait for RustCommutativeMonomial {
    fn adjoint(&self) -> Self {
        Self::new(
            self.data
                .inner_data
                .iter()
                .map(|(operator, &operator_power)| (operator.adjoint(), operator_power))
                .collect(),
            self.data.moment_matrix_id,
        )
    }
}

impl OneWithMomentMatrixId for RustCommutativeMonomial {
    fn one(moment_matrix_id: u8) -> Self {
        Self::new(BTreeMap::new(), moment_matrix_id)
    }
    fn is_one(&self) -> bool {
        self.data.inner_data.is_empty()
    }
}

impl From<&RustCommutativeOperator> for RustCommutativeMonomial {
    fn from(item: &RustCommutativeOperator) -> Self {
        Self::new(BTreeMap::from([(*item, 1)]), item.id.moment_matrix_id)
    }
}

impl From<RustCommutativeOperator> for RustCommutativeMonomial {
    fn from(item: RustCommutativeOperator) -> Self {
        Self::new(BTreeMap::from([(item, 1)]), item.id.moment_matrix_id)
    }
}

impl Mul<&RustCommutativeOperator> for &RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: &RustCommutativeOperator) -> Result<RustCommutativeMonomial, String> {
        rhs * self
    }
}

#[allow(clippy::op_ref)]
impl Mul<RustCommutativeOperator> for &RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: RustCommutativeOperator) -> Result<RustCommutativeMonomial, String> {
        self * &rhs
    }
}

impl Mul<&RustCommutativeOperator> for RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: &RustCommutativeOperator) -> Result<RustCommutativeMonomial, String> {
        rhs * self
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<&RustCommutativeMonomial> for &RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: &RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
            return Err(format!(
                "Cannot multiply monomial {} with moment matrix index {} with monomial {} with moment matrix index {}.",
                self, self.data.moment_matrix_id, rhs, rhs.data.moment_matrix_id
            ));
        }
        Ok(RustCommutativeMonomial::new(
            merge_btreemaps(&self.data.inner_data, &rhs.data.inner_data, |&op, power_left, power_right| {
                if op.id.inner_identifier.is_projector { 1 } else { power_left + power_right }
            }),
            self.data.moment_matrix_id,
        ))
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<RustCommutativeMonomial> for &RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
            return Err(format!(
                "Cannot multiply monomial {} with moment matrix index {} with monomial {} with moment matrix index {}.",
                self, self.data.moment_matrix_id, rhs, rhs.data.moment_matrix_id
            ));
        }
        Ok(RustCommutativeMonomial::new(
            merge_btreemaps(&self.data.inner_data, rhs.data.inner_data, |&op, power_left, power_right| {
                if op.id.inner_identifier.is_projector { 1 } else { power_left + power_right }
            }),
            self.data.moment_matrix_id,
        ))
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<RustCommutativeMonomial> for RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        if self.data.moment_matrix_id != rhs.data.moment_matrix_id {
            return Err(format!(
                "Cannot multiply monomial {} with moment matrix index {} with monomial {} with moment matrix index {}.",
                self, self.data.moment_matrix_id, rhs, rhs.data.moment_matrix_id
            ));
        }
        Ok(RustCommutativeMonomial::new(
            merge_btreemaps(self.data.inner_data, rhs.data.inner_data, |&op, power_left, power_right| {
                if op.id.inner_identifier.is_projector { 1 } else { power_left + power_right }
            }),
            self.data.moment_matrix_id,
        ))
    }
}

impl Mul<&RustCommutativeMonomial> for RustCommutativeMonomial {
    type Output = Result<RustCommutativeMonomial, String>;

    fn mul(self, rhs: &RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        rhs * self
    }
}

impl Mul<&RustRealCoefficientsCommutativePolynomial> for &RustCommutativeMonomial {
    type Output = Result<RustRealCoefficientsCommutativePolynomial, String>;

    fn mul(
        self,
        rhs: &RustRealCoefficientsCommutativePolynomial,
    ) -> Result<RustRealCoefficientsCommutativePolynomial, String> {
        rhs * self
    }
}

impl Mul<&RustComplexCoefficientsCommutativePolynomial> for &RustCommutativeMonomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn mul(
        self,
        rhs: &RustComplexCoefficientsCommutativePolynomial,
    ) -> Result<RustComplexCoefficientsCommutativePolynomial, String> {
        rhs * self
    }
}

#[allow(clippy::op_ref)]
#[cfg(test)]
mod tests {
    use num_complex::Complex;
    use num_traits::{Pow, Zero};
    use rstest::{fixture, rstest};

    use super::*;
    use crate::polynomials::commutative_polynomials::polynomials::commutative_polynomial::{
        RustComplexCoefficientsCommutativePolynomial, RustRealCoefficientsCommutativePolynomial,
    };

    #[fixture]
    fn mon() -> RustCommutativeMonomial {
        RustCommutativeMonomial::new(
            BTreeMap::from([
                (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
            ]),
            0,
        )
    }

    #[rstest]
    fn test_conjugate(mon: RustCommutativeMonomial) {
        let expected = RustCommutativeMonomial::new(
            BTreeMap::from([
                (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
            ]),
            0,
        );
        assert_eq!(mon.adjoint(), expected);
        assert_eq!(mon, expected.adjoint());
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
            (RustCommutativeOperator::new('x', 0, false, false, false, 0), 3),
            (RustCommutativeOperator::new('x', 1, false, false, false, 0), 1),
        ]), 0),
        BTreeMap::from([
            (
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 0, false, false, false, 0), 3),
                ]), 0),
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 1, false, false, false, 0), 1),
                ]), 0),
            ),
            (
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 0, false, false, false, 0), 2),
                    (RustCommutativeOperator::new('x', 1, false, false, false, 0), 1),
                ]), 0),
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 2, false, false, false, 0), 2),
                ]), 0),
            ),
            (
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 0, false, false, false, 0), 1),
                    (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                ]), 0),
                RustCommutativeMonomial::one(0),
            ),
        ]),
        RewritingStrategy::Greedy,
        RustCommutativeMonomial::new(BTreeMap::from([
            (RustCommutativeOperator::new('x', 1, false, false, false, 0), 2),
        ]), 0),
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
            (RustCommutativeOperator::new('x', 0, false, false, false, 0), 3),
            (RustCommutativeOperator::new('x', 1, false, false, false, 0), 1),
        ]), 0),
        BTreeMap::from([
            (
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 0, false, false, false, 0), 4),
                ]), 0),
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 1, false, false, false, 0), 1),
                ]), 0),
            ),
            (
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 0, false, false, false, 0), 2),
                    (RustCommutativeOperator::new('x', 1, false, false, false, 0), 2),
                ]), 0),
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 2, false, false, false, 0), 2),
                ]), 0),
            ),
            (
                RustCommutativeMonomial::new(BTreeMap::from([
                    (RustCommutativeOperator::new('x', 0, false, false, false, 0), 1),
                    (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                ]), 0),
                RustCommutativeMonomial::one(0),
            ),
        ]),
        RewritingStrategy::Greedy,
        initial_mon.clone(),
    )]
    fn test_rewrite(
        #[case] initial_mon: RustCommutativeMonomial,
        #[case] substitution_rules: BTreeMap<RustCommutativeMonomial, RustCommutativeMonomial>,
        #[case] strategy: RewritingStrategy,
        #[case] expected: RustCommutativeMonomial,
    ) {
        assert_eq!(initial_mon.rewrite(strategy, &substitution_rules).unwrap(), expected);
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsCommutativePolynomial::zero())]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(mon.clone(), rhs)]),
        }
    )]
    fn test_mul_complex(
        mon: RustCommutativeMonomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon * rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 2, false, false, false, 0),
        RustCommutativeMonomial::new(BTreeMap::from([
            (RustCommutativeOperator::new('x', 2, false, false, false, 0), 2),
            (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
        ]), 0)
    )]
    #[case(
        RustCommutativeOperator::new('x', 2, true, false, false, 0),
        RustCommutativeMonomial::new(BTreeMap::from([
            (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
            (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
            (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
        ]), 0)
    )]
    fn test_mul_operator(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustCommutativeMonomial,
    ) {
        assert_eq!((&mon * &rhs).unwrap(), expected);
        assert_eq!((&mon * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        mon.clone(),
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
                (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
            ]), 0),
        RustCommutativeMonomial::new(BTreeMap::from([
                (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                (RustCommutativeOperator::new('y', 2, true, false, false, 0), 4),
            ]), 0)
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
                (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
            ]), 0),
        RustCommutativeMonomial::new(BTreeMap::from([
                (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
                (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
            ]), 0)
    )]
    fn test_mul_monomial(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustCommutativeMonomial,
    ) {
        assert_eq!((&mon * &rhs).unwrap(), expected);
        assert_eq!((&mon * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(RustComplexCoefficientsCommutativePolynomial::zero(), RustComplexCoefficientsCommutativePolynomial::zero())]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                        (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
                    ]), 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
                    ]), 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
                    ]), 0),
                    Complex { re: -1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::one(0),
                    Complex { re: 1.2, im: -3.4 },
                ),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, false, false, false, 0), 2),
                        (RustCommutativeOperator::new('y', 2, true, false, false, 0), 4),
                    ]), 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                        (RustCommutativeOperator::new('x', 2, true, false, false, 0), 1),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
                        (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
                    ]), 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, false, false, false, 0), 2),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 2),
                        (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
                    ]), 0),
                    Complex { re: -1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                        (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2),
                    ]), 0),
                    Complex { re: 1.2, im: -3.4 },
                ),
            ]),
        },
    )]
    fn test_mul_polynomial(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&mon * &rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsCommutativePolynomial {
        data: BTreeMap::from([(mon.clone(), Complex { re: 1.0, im: 0.0 })]),
    })]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 1.0, im: 0.0 }),
                (RustCommutativeMonomial::one(0), Complex { re: 1.2, im: 3.4 }),
            ]),
        }
    )]
    fn test_add_complex(
        mon: RustCommutativeMonomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&mon + rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsCommutativePolynomial {
        data: BTreeMap::from([(mon.clone(), Complex { re: 1.0, im: 0.0 })]),
    })]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 1.0, im: 0.0 }),
                (RustCommutativeMonomial::one(0), Complex { re: -1.2, im: -3.4 }),
            ]),
        }
    )]
    fn test_sub_complex(
        mon: RustCommutativeMonomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&mon - rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 1, false, false, false, 0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), 1.0),
                (RustCommutativeMonomial::from(rhs), 1.0),
            ]),
        }
    )]
    fn test_add_operator(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustRealCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon + &rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 1, false, false, false, 0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), 1.0),
                (RustCommutativeMonomial::from(rhs), -1.0),
            ]),
        }
    )]
    fn test_sub_operator(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustRealCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon - &rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), 1.0),
                (RustCommutativeMonomial::one(0), 1.0),
            ]),
        }
    )]
    #[case(
        mon.clone(),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), 2.0),
            ]),
        }
    )]
    fn test_add_monomial(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustRealCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon + &rhs, expected);
        assert_eq!(&mon + rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), 1.0),
                (RustCommutativeMonomial::one(0), -1.0),
            ]),
        }
    )]
    #[case(
        mon.clone(),
        RustRealCoefficientsCommutativePolynomial::zero()
    )]
    fn test_sub_monomial(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustRealCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon - &rhs, expected);
        assert_eq!(&mon - rhs, expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsCommutativePolynomial::zero(),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(mon.clone(), Complex { re: 1.0, im: 0.0 })]),
        },
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 1.5, im: 3.4 }),
                (RustCommutativeMonomial::one(0), Complex { re: -5.3, im: 5.2 }),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 2.5, im: 3.4 }),
                (RustCommutativeMonomial::one(0), Complex { re: -5.3, im: 5.2 }),
            ]),
        },
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: -5.3, im: 5.2 }),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 1.0, im: 0.0 }),
                (RustCommutativeMonomial::one(0), Complex { re: -5.3, im: 5.2 }),
            ]),
        },
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(mon.clone(), Complex { re: -1.0, im: 0.0 })]),
        },
        RustComplexCoefficientsCommutativePolynomial::zero(),
    )]
    fn test_add_polynomial(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon + &rhs, expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsCommutativePolynomial::zero(),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(mon.clone(), Complex { re: 1.0, im: 0.0 })]),
        },
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 1.5, im: 3.4 }),
                (RustCommutativeMonomial::one(0), Complex { re: -5.3, im: 5.2 }),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: -0.5, im: -3.4 }),
                (RustCommutativeMonomial::one(0), Complex { re: 5.3, im: -5.2 }),
            ]),
        },
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: -5.3, im: 5.2 }),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (mon.clone(), Complex { re: 1.0, im: 0.0 }),
                (RustCommutativeMonomial::one(0), Complex { re: 5.3, im: -5.2 }),
            ]),
        },
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(mon.clone(), Complex { re: 1.0, im: 0.0 })]),
        },
        RustComplexCoefficientsCommutativePolynomial::zero(),
    )]
    fn test_sub_polynomial(
        mon: RustCommutativeMonomial,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&mon - &rhs, expected);
    }

    #[rstest]
    fn test_neg(mon: RustCommutativeMonomial) {
        let expected = RustRealCoefficientsCommutativePolynomial { data: BTreeMap::from([(mon.clone(), -1.0)]) };
        assert_eq!(-&mon, expected);
        assert_eq!(-mon, expected);
    }

    #[rstest]
    fn test_pow(mon: RustCommutativeMonomial) {
        let intended_result = RustCommutativeMonomial::one(0);
        assert_eq!(mon.pow(0).unwrap(), intended_result);

        for power in [1u8, 2u8] {
            let intended_result = RustCommutativeMonomial::new(
                BTreeMap::from([
                    (RustCommutativeOperator::new('x', 2, false, false, false, 0), power),
                    (RustCommutativeOperator::new('y', 2, true, false, false, 0), 2 * power),
                ]),
                0,
            );
            assert_eq!(mon.pow(power).unwrap(), intended_result);
        }
    }

    #[rstest]
    fn test_try_mul_monomial_different_party(mon: RustCommutativeMonomial) {
        let op_party1 = RustCommutativeOperator::new('x', 5, false, false, false, 1);
        let mon_party1 = RustCommutativeMonomial::from(op_party1);
        assert!((&mon * &mon_party1).is_err());
    }
}
