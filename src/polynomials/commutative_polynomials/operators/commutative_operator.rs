use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hasher;
use std::ops::{Add, Mul, Neg, Sub};

use num_complex::Complex;
use num_traits::Pow;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyNone;
use rustc_hash::FxHasher;

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::{
    PythonCommutativeMonomial, RustCommutativeMonomial,
};
use crate::polynomials::commutative_polynomials::polynomials::commutative_polynomial::{
    PythonComplexCoefficientsCommutativePolynomial, PythonRealCoefficientsCommutativePolynomial,
    RustCommutativePolynomial,
};
use crate::polynomials::monomial::{AdjointTrait, HasAMomentMatrixId, OneWithMomentMatrixId};
use crate::polynomials::operator::Operator;
use crate::polynomials::polynomial::PolynomialDtype;
use crate::polynomials::utils::add::manage_entry;
use crate::relaxations::constraint::{ConstraintKind, make_commutative_constraint};

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct CommutativeOperatorIdentifier {
    index: u8,
    label: char, // We could convert to an Arc<str>, but we lose in performance by doing so
    is_adjoint: bool,
    is_hermitian: bool,
    pub(crate) is_projector: bool,
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct CommutativeOperatorIdentifierWithMomentMatrixIndex {
    pub(crate) inner_identifier: CommutativeOperatorIdentifier,
    pub(crate) moment_matrix_id: u8,
}

impl HasAMomentMatrixId for CommutativeOperatorIdentifierWithMomentMatrixIndex {
    fn moment_matrix_id(&self) -> u8 {
        self.moment_matrix_id
    }
}

pub(crate) type RustCommutativeOperator = Operator<CommutativeOperatorIdentifierWithMomentMatrixIndex>;

/// A single commutative scalar variable (operator).
///
/// Commutative operators represent ordinary scalar variables that commute
/// with each other (i.e. `x * y == y * x`).  They are the building blocks
/// for constructing [`CommutativeMonomial`] objects and commutative polynomial
/// expressions.
///
/// Instances are normally created in bulk via
/// [`generate_commutative_variables`].
#[pyclass(frozen, module = "ncpoleon.polynomials.commutative_polynomials", name = "CommutativeOperator")]
#[derive(Clone, Copy)]
pub(crate) struct PythonCommutativeOperator(pub(crate) RustCommutativeOperator);

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonCommutativeOperator {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(mon) = value.cast::<PythonCommutativeOperator>() {
            Ok(*mon.get())
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonCommutativeOperator"))
        }
    }
}

#[pymethods]
impl PythonCommutativeOperator {
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
            PythonCommutativeMonomial((&self.0 * &op.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            PythonCommutativeMonomial((self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 * &poly_real.get().0).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
            PythonComplexCoefficientsCommutativePolynomial(
                (&self.0 * &poly_complex.get().0).map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial(self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial(self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        // __rmul__ is only called if __mul__ isn't supported on other, which means that it's either
        // a scalar, or not supported
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial(self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial(self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        // Can't divide by anything else than a scalar for now
        if let Ok(lambda_real) = other.extract::<f64>() {
            PythonRealCoefficientsCommutativePolynomial(self.0 * (1.0 / lambda_real)).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsCommutativePolynomial(self.0 * (1.0 / lambda_complex)).into_py_any(py)
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

    fn __pow__<'py>(&self, power: u8, _modulo: &Bound<'py, PyNone>) -> PythonCommutativeMonomial {
        PythonCommutativeMonomial((&self.0).pow(power))
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = (*self).into_py_any(py)?;
        make_commutative_constraint(self_any.bind(py), other, ConstraintKind::Equality)
    }

    fn __ge__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = (*self).into_py_any(py)?;
        make_commutative_constraint(self_any.bind(py), other, ConstraintKind::Inequality)
    }

    fn __le__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = (*self).into_py_any(py)?;
        make_commutative_constraint(other, self_any.bind(py), ConstraintKind::Inequality)
    }

    pub(crate) fn __hash__(&self) -> u64 {
        self.0.__hash__()
    }

    #[getter]
    fn moment_matrix_id(&self) -> u8 {
        self.0.id.moment_matrix_id
    }

    /// Return the complex conjugate of this operator.
    pub(crate) fn adjoint(&self) -> PythonCommutativeOperator {
        Self(self.0.adjoint())
    }
}

impl RustCommutativeOperator {
    pub(crate) fn new(
        label: char,
        index: u8,
        is_adjoint: bool,
        is_hermitian: bool,
        is_projector: bool,
        moment_matrix_id: u8,
    ) -> Self {
        Self {
            id: CommutativeOperatorIdentifierWithMomentMatrixIndex {
                inner_identifier: CommutativeOperatorIdentifier {
                    index,
                    label,
                    is_adjoint,
                    is_hermitian,
                    is_projector,
                },
                moment_matrix_id,
            },
        }
    }

    pub(crate) fn __str__(&self) -> String {
        self.to_string()
    }

    pub(crate) fn __hash__(&self) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_u32(self.id.inner_identifier.label as u32);
        hasher.write_u8(self.id.inner_identifier.index);
        hasher.write_u8(self.id.inner_identifier.is_adjoint as u8);
        hasher.write_u8(self.id.moment_matrix_id);
        hasher.finish()
    }
}

impl fmt::Display for RustCommutativeOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.id.inner_identifier.is_adjoint {
            write!(f, "{}_({})^*", self.id.inner_identifier.label, self.id.inner_identifier.index)
        } else {
            write!(f, "{}_({})", self.id.inner_identifier.label, self.id.inner_identifier.index)
        }
    }
}

impl AdjointTrait for RustCommutativeOperator {
    fn adjoint(&self) -> Self {
        if self.id.inner_identifier.is_hermitian {
            *self
        } else {
            Self::new(
                self.id.inner_identifier.label,
                self.id.inner_identifier.index,
                !self.id.inner_identifier.is_adjoint,
                false,
                false,
                self.id.moment_matrix_id,
            )
        }
    }
}

impl<Scalar: PolynomialDtype> Mul<Scalar> for &RustCommutativeOperator {
    type Output = RustCommutativePolynomial<Scalar>;
    fn mul(self, rhs: Scalar) -> RustCommutativePolynomial<Scalar> {
        RustCommutativeMonomial::from(self) * rhs
    }
}

impl<Scalar: PolynomialDtype> Mul<Scalar> for RustCommutativeOperator {
    type Output = RustCommutativePolynomial<Scalar>;
    fn mul(self, rhs: Scalar) -> RustCommutativePolynomial<Scalar> {
        RustCommutativeMonomial::from(self) * rhs
    }
}

impl Mul<&RustCommutativeOperator> for &RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, rhs: &RustCommutativeOperator) -> Result<RustCommutativeMonomial, String> {
        if self.id.moment_matrix_id != rhs.id.moment_matrix_id {
            return Err(format!(
                "Cannot multiply operator {} with moment matrix index {} with monomial {} with moment matrix index {}.",
                self, self.id.moment_matrix_id, rhs, rhs.id.moment_matrix_id
            ));
        }
        Ok(if self == rhs {
            if self.id.inner_identifier.is_projector {
                RustCommutativeMonomial::new(BTreeMap::from([(*self, 1)]), self.id.moment_matrix_id)
            } else {
                RustCommutativeMonomial::new(BTreeMap::from([(*self, 2)]), self.id.moment_matrix_id)
            }
        } else {
            RustCommutativeMonomial::new(BTreeMap::from([(*self, 1), (*rhs, 1)]), self.id.moment_matrix_id)
        })
    }
}

#[allow(clippy::op_ref)]
impl Mul<&RustCommutativeOperator> for RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, rhs: &RustCommutativeOperator) -> Result<RustCommutativeMonomial, String> {
        &self * rhs
    }
}

#[allow(clippy::op_ref)]
impl Mul<RustCommutativeOperator> for &RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, rhs: RustCommutativeOperator) -> Result<RustCommutativeMonomial, String> {
        self * &rhs
    }
}

impl Pow<u8> for &RustCommutativeOperator {
    type Output = RustCommutativeMonomial;

    fn pow(self, rhs: u8) -> Self::Output {
        if rhs == 0 {
            return RustCommutativeMonomial::one(self.id.moment_matrix_id);
        }
        if self.id.inner_identifier.is_projector {
            return RustCommutativeMonomial::from(self);
        }

        RustCommutativeMonomial::new(BTreeMap::from([(*self, rhs)]), self.id.moment_matrix_id)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<&RustCommutativeMonomial> for &RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, rhs: &RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        if self.id.moment_matrix_id != rhs.data.moment_matrix_id {
            return Err(format!(
                "Cannot multiply operator {} with moment matrix index {} with monomial {} with moment matrix index {}.",
                self, self.id.moment_matrix_id, rhs, rhs.data.moment_matrix_id
            ));
        }

        let mut degrees = rhs.data.inner_data.clone();
        if self.id.inner_identifier.is_projector {
            degrees.entry(*self).or_insert(1);
        } else {
            degrees.entry(*self).and_modify(|power| *power += 1).or_insert(1);
        }
        Ok(RustCommutativeMonomial::new(degrees, self.id.moment_matrix_id))
    }
}

#[allow(clippy::op_ref)]
impl Mul<&RustCommutativeMonomial> for RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, rhs: &RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        &self * rhs
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul<RustCommutativeMonomial> for &RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, mut rhs: RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        if self.id.moment_matrix_id != rhs.data.moment_matrix_id {
            return Err(format!(
                "Cannot multiply operator {} with moment matrix index {} with monomial {} with moment matrix index {}.",
                self, self.id.moment_matrix_id, rhs, rhs.data.moment_matrix_id
            ));
        }
        let entry = rhs.data.inner_data.entry(*self);

        if self.id.inner_identifier.is_projector {
            entry.or_insert(1);
        } else {
            entry.and_modify(|power| *power += 1).or_insert(1);
        }

        Ok(rhs)
    }
}

#[allow(clippy::op_ref)]
impl Mul<RustCommutativeMonomial> for RustCommutativeOperator {
    type Output = Result<RustCommutativeMonomial, String>;
    fn mul(self, rhs: RustCommutativeMonomial) -> Result<RustCommutativeMonomial, String> {
        &self * rhs
    }
}

impl<Scalar: PolynomialDtype> Mul<&RustCommutativePolynomial<Scalar>> for &RustCommutativeOperator {
    type Output = Result<RustCommutativePolynomial<Scalar>, String>;
    fn mul(self, rhs: &RustCommutativePolynomial<Scalar>) -> Result<RustCommutativePolynomial<Scalar>, String> {
        let mut res = BTreeMap::new();

        for (mon, &coeff) in rhs.data.iter() {
            manage_entry(&mut res, (self * mon)?, coeff);
        }

        Ok(RustCommutativePolynomial { data: res })
    }
}

impl<Scalar: PolynomialDtype> Add<Scalar> for &RustCommutativeOperator {
    type Output = Result<RustCommutativePolynomial<Scalar>, String>;
    fn add(self, rhs: Scalar) -> Result<RustCommutativePolynomial<Scalar>, String> {
        RustCommutativeMonomial::from(self) + rhs
    }
}

impl<Scalar: PolynomialDtype> Sub<Scalar> for &RustCommutativeOperator {
    type Output = Result<RustCommutativePolynomial<Scalar>, String>;
    fn sub(self, rhs: Scalar) -> Result<RustCommutativePolynomial<Scalar>, String> {
        RustCommutativeMonomial::from(self) - rhs
    }
}

impl Add<&RustCommutativeOperator> for &RustCommutativeOperator {
    type Output = RustCommutativePolynomial<f64>;
    fn add(self, rhs: &RustCommutativeOperator) -> RustCommutativePolynomial<f64> {
        RustCommutativeMonomial::from(self) + RustCommutativeMonomial::from(rhs)
    }
}

impl Sub<&RustCommutativeOperator> for &RustCommutativeOperator {
    type Output = RustCommutativePolynomial<f64>;
    fn sub(self, rhs: &RustCommutativeOperator) -> RustCommutativePolynomial<f64> {
        RustCommutativeMonomial::from(self) - RustCommutativeMonomial::from(rhs)
    }
}

#[allow(clippy::op_ref)]
impl Add<RustCommutativeOperator> for &RustCommutativeOperator {
    type Output = RustCommutativePolynomial<f64>;
    fn add(self, rhs: RustCommutativeOperator) -> RustCommutativePolynomial<f64> {
        self + &rhs
    }
}

#[allow(clippy::op_ref)]
impl Sub<RustCommutativeOperator> for &RustCommutativeOperator {
    type Output = RustCommutativePolynomial<f64>;
    fn sub(self, rhs: RustCommutativeOperator) -> RustCommutativePolynomial<f64> {
        self - &rhs
    }
}

impl Neg for RustCommutativeOperator {
    type Output = RustCommutativePolynomial<f64>;

    fn neg(self) -> RustCommutativePolynomial<f64> {
        -RustCommutativeMonomial::from(self)
    }
}

impl Neg for &RustCommutativeOperator {
    type Output = RustCommutativePolynomial<f64>;

    fn neg(self) -> RustCommutativePolynomial<f64> {
        -RustCommutativeMonomial::from(self)
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
    fn op() -> RustCommutativeOperator {
        RustCommutativeOperator::new('x', 2, false, false, false, 0)
    }

    #[rstest]
    fn test_conjugate(op: RustCommutativeOperator) {
        let expected = RustCommutativeOperator::new('x', 2, true, false, false, 0);
        assert_eq!(op.adjoint(), expected);
        assert_eq!(op, expected.adjoint());
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsCommutativePolynomial::zero())]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(RustCommutativeMonomial::from(op), rhs)]),
        }
    )]
    fn test_mul_complex(
        op: RustCommutativeOperator,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&op * rhs, expected);
        assert_eq!(op * rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 2, true, false, false, 0),
        RustCommutativeMonomial::new(BTreeMap::from([(op, 1), (rhs, 1)]), 0),
    )]
    #[case(
        RustCommutativeOperator::new('x', 2, false, false, false, 0),
        RustCommutativeMonomial::new(BTreeMap::from([(rhs, 2)]), 0),
    )]
    fn test_mul_operator(
        op: RustCommutativeOperator,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustCommutativeMonomial,
    ) {
        assert_eq!((&op * &rhs).unwrap(), expected);
        assert_eq!((op * &rhs).unwrap(), expected);
        assert_eq!((&op * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        RustCommutativeMonomial::new(BTreeMap::from([(op, 1)]), 0),
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
            (op, 2),
            (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
        ]), 0),
        RustCommutativeMonomial::new(BTreeMap::from([
            (op, 3),
            (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
        ]), 0),
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
            (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
        ]), 0),
        RustCommutativeMonomial::new(BTreeMap::from([
            (op, 1),
            (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
        ]), 0),
    )]
    fn test_mul_monomial(
        op: RustCommutativeOperator,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustCommutativeMonomial,
    ) {
        assert_eq!((&op * &rhs).unwrap(), expected);
        assert_eq!((op * &rhs).unwrap(), expected);
        assert_eq!((op * rhs.clone()).unwrap(), expected);
        assert_eq!((&op * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(RustComplexCoefficientsCommutativePolynomial::zero(), RustComplexCoefficientsCommutativePolynomial::zero())]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (op, 2),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
                    ]), 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
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
                        (op, 3),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
                    ]), 0),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                        (op, 1),
                        (RustCommutativeOperator::new('y', 2, false, false, false, 0), 3),
                    ]), 0),
                    Complex { re: -1.2, im: 3.4 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(op, 1)]), 0),
                    Complex { re: 1.2, im: -3.4 },
                ),
            ]),
        },
    )]
    fn test_mul_polynomial(
        op: RustCommutativeOperator,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&op * &rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex::ZERO,
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(RustCommutativeMonomial::from(op), Complex { re: 1.0, im: 0.0 })]),
        },
    )]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::from(op), Complex { re: 1.0, im: 0.0 }),
                (RustCommutativeMonomial::one(0), Complex { re: 1.2, im: 3.4 }),
            ]),
        },
    )]
    fn test_add_complex(
        op: RustCommutativeOperator,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&op + rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex::ZERO,
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(RustCommutativeMonomial::from(op), Complex { re: 1.0, im: 0.0 })]),
        },
    )]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::from(op), Complex { re: 1.0, im: 0.0 }),
                (RustCommutativeMonomial::one(0), Complex { re: -1.2, im: -3.4 }),
            ]),
        },
    )]
    fn test_sub_complex(
        op: RustCommutativeOperator,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&op - rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 1, false, false, false, 0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::from(op), 1.0),
                (RustCommutativeMonomial::from(rhs), 1.0),
            ]),
        }
    )]
    #[case(
        RustCommutativeOperator::new('x', 2, false, false, false, 0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(RustCommutativeMonomial::from(rhs), 2.0)]),
        }
    )]
    fn test_add_operator(
        op: RustCommutativeOperator,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustRealCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&op + &rhs, expected);
        assert_eq!(&op + rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 1, false, false, false, 0),
        RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::from(op), 1.0),
                (RustCommutativeMonomial::from(rhs), -1.0),
            ]),
        }
    )]
    #[case(
        RustCommutativeOperator::new('x', 2, false, false, false, 0),
        RustRealCoefficientsCommutativePolynomial::zero()
    )]
    fn test_sub_operator(
        op: RustCommutativeOperator,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustRealCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&op - &rhs, expected);
        assert_eq!(&op - rhs, expected);
    }

    #[rstest]
    fn test_neg(op: RustCommutativeOperator) {
        let expected = RustRealCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(RustCommutativeMonomial::from(op), -1.0)]),
        };
        assert_eq!(-&op, expected);
        assert_eq!(-op, expected);
    }

    #[rstest]
    fn test_pow(op: RustCommutativeOperator) {
        let intended_result = RustCommutativeMonomial::one(0);
        assert_eq!((&op).pow(0u8), intended_result);

        for power in [1u8, 2u8] {
            let intended_result = RustCommutativeMonomial::new(BTreeMap::from([(op, power)]), 0);
            assert_eq!((&op).pow(power), intended_result);
        }
    }

    #[rstest]
    fn test_try_mul_operator_different_party(op: RustCommutativeOperator) {
        let op_party1 = RustCommutativeOperator::new('x', 2, false, false, false, 1);
        assert!((&op * &op_party1).is_err());
    }

    #[rstest]
    fn test_try_mul_monomial_different_party(op: RustCommutativeOperator) {
        let op_party1 = RustCommutativeOperator::new('x', 3, false, false, false, 1);
        let mon_party1 = RustCommutativeMonomial::from(op_party1);
        assert!((&op * &mon_party1).is_err());
    }
}
