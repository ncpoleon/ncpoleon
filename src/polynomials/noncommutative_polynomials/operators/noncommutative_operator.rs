use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hasher;
use std::iter::once;
use std::ops::{Add, Mul, Neg, Sub};

use num_complex::Complex;
use num_traits::Pow;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyNone;
use rustc_hash::FxHasher;

use crate::polynomials::monomial::{AdjointTrait, HasAMomentMatrixId, OneWithMomentMatrixId};
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::{
    PythonNonCommutativeMonomial, RustNonCommutativeMonomial,
};
use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::{
    PythonComplexCoefficientsNonCommutativePolynomial, PythonRealCoefficientsNonCommutativePolynomial,
    RustNonCommutativePolynomial,
};
use crate::polynomials::operator::Operator;
use crate::polynomials::polynomial::PolynomialDtype;
use crate::polynomials::utils::add::manage_entry;
use crate::relaxations::constraint::{ConstraintKind, make_noncommutative_constraint};

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct NonCommutativeOperatorIdentifier {
    pub(crate) index: u8,
    pub(crate) label: char, /* We could convert to an Arc<str>, but we lose in performance by
                             * doing so */
    pub(crate) is_adjoint: bool,
    pub(crate) is_hermitian: bool,
    pub(crate) is_projector: bool,
    pub(crate) moment_matrix_id: u8,
}

impl HasAMomentMatrixId for NonCommutativeOperatorIdentifier {
    fn moment_matrix_id(&self) -> u8 {
        self.moment_matrix_id
    }
}

pub(crate) type RustNonCommutativeOperator = Operator<NonCommutativeOperatorIdentifier>;

/// A single non-commutative operator (variable).
///
/// Non-commutative operators do **not** commute in general: `A * B ≠ B * A`.
/// They are the building blocks for [`NonCommutativeMonomial`] and noncommutative
/// polynomial expressions with real or complex coefficients.
///
/// Instances are normally created in bulk via
/// [`generate_noncommutative_variables`].
#[pyclass(frozen, module = "ncpoleon.polynomials.noncommutative_polynomials", name = "NonCommutativeOperator")]
#[derive(Clone, Copy)]
pub(crate) struct PythonNonCommutativeOperator(pub(crate) RustNonCommutativeOperator);

#[pymethods]
impl PythonNonCommutativeOperator {
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
            PythonNonCommutativeMonomial((&self.0 * &op.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            PythonNonCommutativeMonomial((self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
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
        Ok(PythonNonCommutativeMonomial((&self.0).pow(power).map_err(PyValueError::new_err)?))
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = (*self).into_py_any(py)?;
        make_noncommutative_constraint(self_any.bind(py), other, ConstraintKind::Equality)
    }

    fn __ge__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = (*self).into_py_any(py)?;
        make_noncommutative_constraint(self_any.bind(py), other, ConstraintKind::Inequality)
    }

    fn __le__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let self_any = (*self).into_py_any(py)?;
        make_noncommutative_constraint(other, self_any.bind(py), ConstraintKind::Inequality)
    }

    pub(crate) fn __hash__(&self) -> u64 {
        self.0.__hash__()
    }

    #[getter]
    fn moment_matrix_id(&self) -> u8 {
        self.0.id.moment_matrix_id
    }

    /// Return the adjoint (Hermitian conjugate) of this operator.
    pub(crate) fn adjoint(&self) -> PythonNonCommutativeOperator {
        Self(self.0.adjoint())
    }
}

impl RustNonCommutativeOperator {
    pub(crate) fn new(
        label: char,
        index: u8,
        is_adjoint: bool,
        is_hermitian: bool,
        is_projector: bool,
        party: u8,
    ) -> Self {
        Self {
            id: NonCommutativeOperatorIdentifier {
                index,
                label,
                is_adjoint,
                is_hermitian,
                is_projector,
                moment_matrix_id: party,
            },
        }
    }

    pub(crate) fn __str__(&self) -> String {
        self.to_string()
    }

    pub(crate) fn __hash__(&self) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write_u8(self.id.index);
        hasher.write_u8(self.id.label as u8);
        hasher.write_u8(self.id.is_adjoint as u8);
        hasher.write_u8(self.id.moment_matrix_id);
        hasher.finish()
    }
}

impl fmt::Display for RustNonCommutativeOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.id.is_adjoint {
            write!(f, "{}({})^†", self.id.label, self.id.index)
        } else {
            write!(f, "{}({})", self.id.label, self.id.index)
        }
    }
}

impl AdjointTrait for RustNonCommutativeOperator {
    fn adjoint(&self) -> Self {
        if self.id.is_hermitian {
            *self
        } else {
            Self::new(
                self.id.label,
                self.id.index,
                !self.id.is_adjoint,
                false,
                false, // Can't be a projector if it's not Hermitian
                self.id.moment_matrix_id,
            )
        }
    }
}

impl<Scalar: PolynomialDtype> Mul<Scalar> for &RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<Scalar>;

    fn mul(self, rhs: Scalar) -> Self::Output {
        RustNonCommutativeMonomial::from(self) * rhs
    }
}

impl Mul<&RustNonCommutativeOperator> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, rhs: &RustNonCommutativeOperator) -> Self::Output {
        if self.id.moment_matrix_id != rhs.id.moment_matrix_id {
            return Err(format!(
                "Cannot multiply operators from different parties: {} and {}",
                self.id.moment_matrix_id, rhs.id.moment_matrix_id
            ));
        }
        Ok(if (self == rhs) & self.id.is_projector {
            RustNonCommutativeMonomial::new(vec![*self], self.id.moment_matrix_id)
        } else {
            RustNonCommutativeMonomial::new(vec![*self, *rhs], self.id.moment_matrix_id)
        })
    }
}

#[allow(clippy::op_ref)]
impl Mul<&RustNonCommutativeOperator> for RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, rhs: &RustNonCommutativeOperator) -> Self::Output {
        &self * rhs
    }
}

#[allow(clippy::op_ref)]
impl Mul<RustNonCommutativeOperator> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, rhs: RustNonCommutativeOperator) -> Self::Output {
        self * &rhs
    }
}

impl Pow<u8> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;

    fn pow(self, rhs: u8) -> Self::Output {
        if rhs == 0 {
            return Ok(RustNonCommutativeMonomial::one(self.id.moment_matrix_id));
        }
        if self.id.is_projector {
            return Ok(RustNonCommutativeMonomial::from(self));
        }
        Ok(RustNonCommutativeMonomial::new(vec![*self; rhs as usize], self.id.moment_matrix_id))
    }
}

impl Mul<&RustNonCommutativeMonomial> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, rhs: &RustNonCommutativeMonomial) -> Self::Output {
        if let Some(first) = rhs.data.inner_data.first() {
            if self.id.moment_matrix_id != first.id.moment_matrix_id {
                return Err(format!(
                    "Cannot multiply operators from different parties: {} and {}",
                    self.id.moment_matrix_id, first.id.moment_matrix_id
                ));
            }
            Ok(if (first == self) & self.id.is_projector {
                rhs.clone()
            } else {
                let mut res = Vec::with_capacity(1 + rhs.data.inner_data.len());
                res.push(*self);
                res.extend(rhs.data.inner_data.to_vec());
                RustNonCommutativeMonomial::new(res, self.id.moment_matrix_id)
            })
        } else {
            if self.id.moment_matrix_id != rhs.data.moment_matrix_id {
                return Err(format!(
                    "Cannot multiply operators from different parties: {} and {}",
                    self.id.moment_matrix_id, rhs.data.moment_matrix_id
                ));
            }
            Ok(RustNonCommutativeMonomial::from(self))
        }
    }
}

#[allow(clippy::op_ref)]
impl Mul<&RustNonCommutativeMonomial> for RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, rhs: &RustNonCommutativeMonomial) -> Self::Output {
        &self * rhs
    }
}

impl Mul<RustNonCommutativeMonomial> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, mut rhs: RustNonCommutativeMonomial) -> Self::Output {
        if let Some(first) = rhs.data.inner_data.first().copied() {
            if self.id.moment_matrix_id != first.id.moment_matrix_id {
                return Err(format!(
                    "Cannot multiply operators from different parties: {} and {}",
                    self.id.moment_matrix_id, first.id.moment_matrix_id
                ));
            }
            if (first != *self) | !self.id.is_projector {
                rhs.data.inner_data.splice(0..0, once(*self));
            }
            Ok(rhs)
        } else {
            if self.id.moment_matrix_id != rhs.data.moment_matrix_id {
                return Err(format!(
                    "Cannot multiply operators from different parties: {} and {}",
                    self.id.moment_matrix_id, rhs.data.moment_matrix_id
                ));
            }
            Ok(RustNonCommutativeMonomial::from(self))
        }
    }
}

#[allow(clippy::op_ref)]
impl Mul<RustNonCommutativeMonomial> for RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativeMonomial, String>;
    fn mul(self, rhs: RustNonCommutativeMonomial) -> Self::Output {
        &self * rhs
    }
}

impl<Scalar: PolynomialDtype> Mul<&RustNonCommutativePolynomial<Scalar>> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativePolynomial<Scalar>, String>;
    fn mul(self, rhs: &RustNonCommutativePolynomial<Scalar>) -> Self::Output {
        let mut res = BTreeMap::new();

        for (mon, &coeff) in rhs.data.iter() {
            manage_entry(&mut res, (self * mon)?, coeff);
        }

        Ok(RustNonCommutativePolynomial { data: res })
    }
}

impl<Scalar: PolynomialDtype> Add<Scalar> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativePolynomial<Scalar>, String>;
    fn add(self, rhs: Scalar) -> Self::Output {
        RustNonCommutativeMonomial::from(self) + rhs
    }
}

impl<Scalar: PolynomialDtype> Sub<Scalar> for &RustNonCommutativeOperator {
    type Output = Result<RustNonCommutativePolynomial<Scalar>, String>;
    fn sub(self, rhs: Scalar) -> Self::Output {
        RustNonCommutativeMonomial::from(self) - rhs
    }
}

impl Add<&RustNonCommutativeOperator> for &RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<f64>;
    fn add(self, rhs: &RustNonCommutativeOperator) -> RustNonCommutativePolynomial<f64> {
        RustNonCommutativeMonomial::from(self) + RustNonCommutativeMonomial::from(rhs)
    }
}

impl Sub<&RustNonCommutativeOperator> for &RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<f64>;
    fn sub(self, rhs: &RustNonCommutativeOperator) -> RustNonCommutativePolynomial<f64> {
        RustNonCommutativeMonomial::from(self) - RustNonCommutativeMonomial::from(rhs)
    }
}

#[allow(clippy::op_ref)]
impl Add<RustNonCommutativeOperator> for &RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<f64>;
    fn add(self, rhs: RustNonCommutativeOperator) -> RustNonCommutativePolynomial<f64> {
        self + &rhs
    }
}

#[allow(clippy::op_ref)]
impl Sub<RustNonCommutativeOperator> for &RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<f64>;
    fn sub(self, rhs: RustNonCommutativeOperator) -> RustNonCommutativePolynomial<f64> {
        self - &rhs
    }
}

impl Neg for RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<f64>;

    fn neg(self) -> RustNonCommutativePolynomial<f64> {
        -RustNonCommutativeMonomial::from(self)
    }
}

impl Neg for &RustNonCommutativeOperator {
    type Output = RustNonCommutativePolynomial<f64>;

    fn neg(self) -> RustNonCommutativePolynomial<f64> {
        -RustNonCommutativeMonomial::from(self)
    }
}

#[allow(clippy::op_ref)]
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use num_complex::Complex;
    use num_traits::Zero;
    use rstest::{fixture, rstest};

    use super::*;
    use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::RustNonCommutativeMonomial;
    use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::RustComplexCoefficientsNonCommutativePolynomial;

    #[fixture]
    fn op() -> RustNonCommutativeOperator {
        RustNonCommutativeOperator::new('x', 2, false, false, false, 0)
    }

    #[fixture]
    fn op_herm() -> RustNonCommutativeOperator {
        RustNonCommutativeOperator::new('x', 2, false, true, false, 0)
    }

    #[fixture]
    fn op_proj() -> RustNonCommutativeOperator {
        RustNonCommutativeOperator::new('x', 2, false, true, true, 0)
    }

    #[rstest]
    fn test_adjoint_op(op: RustNonCommutativeOperator) {
        let expected = RustNonCommutativeOperator::new('x', 2, true, false, false, 0);
        assert_eq!(op.adjoint(), expected);
        assert_eq!(op, expected.adjoint());
    }

    #[rstest]
    fn test_adjoint_op_hermitian(op_herm: RustNonCommutativeOperator) {
        let expected = RustNonCommutativeOperator::new('x', 2, false, true, false, 0);
        assert_eq!(op_herm.adjoint(), expected);
        assert_eq!(op_herm, expected.adjoint());
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsNonCommutativePolynomial::zero())]
    #[case(
        Complex { re: 1.2, im: 3.4 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([(
                RustNonCommutativeMonomial::from(op),
                rhs,
            )]),
        }
    )]
    fn test_mul_complex(
        op: RustNonCommutativeOperator,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&op * rhs, expected);
    }

    #[rstest]
    #[case(
        op,
        RustNonCommutativeMonomial::new(vec![op, op], 0),
    )]
    #[case(
        RustNonCommutativeOperator::new(
            'x',
            1,
            false,
            false,
            false,
            0
        ),
        RustNonCommutativeMonomial::new(vec![op, rhs], 0),
    )]
    fn test_mul_operator(
        op: RustNonCommutativeOperator,
        #[case] rhs: RustNonCommutativeOperator,
        #[case] expected: RustNonCommutativeMonomial,
    ) {
        assert_eq!((&op * &rhs).unwrap(), expected);
        assert_eq!((&op * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        op_proj,
        RustNonCommutativeMonomial::new(vec![op_proj], 0),
    )]
    #[case(
        RustNonCommutativeOperator::new(
            'x',
            1,
            false,
            true,
            true,
            0
        ),
        RustNonCommutativeMonomial::new(vec![op_proj, rhs], 0),
    )]
    fn test_mul_operator_proj(
        op_proj: RustNonCommutativeOperator,
        #[case] rhs: RustNonCommutativeOperator,
        #[case] expected: RustNonCommutativeMonomial,
    ) {
        assert_eq!((&op_proj * &rhs).unwrap(), expected);
        assert_eq!((&op_proj * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(RustNonCommutativeMonomial::one(0), RustNonCommutativeMonomial::from(op))]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            op,
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
        ], 0),
        RustNonCommutativeMonomial::new(vec![
            op,
            op,
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
        ], 0),
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
            op,
        ], 0),
        RustNonCommutativeMonomial::new(vec![
            op,
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
            op
        ], 0),
    )]
    fn test_mul_monomial(
        op: RustNonCommutativeOperator,
        #[case] rhs: RustNonCommutativeMonomial,
        #[case] expected: RustNonCommutativeMonomial,
    ) {
        assert_eq!((&op * &rhs).unwrap(), expected);
        assert_eq!((&op * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(RustNonCommutativeMonomial::one(0), RustNonCommutativeMonomial::from(op_proj))]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            op_proj,
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
        ], 0),
        RustNonCommutativeMonomial::new(vec![
            op_proj,
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
        ], 0),
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
            op_proj,
        ], 0),
        RustNonCommutativeMonomial::new(vec![
            op_proj,
            RustNonCommutativeOperator::new(
                'x',
                1,
                false,
                false,
                false,
                0
            ),
            op_proj
        ], 0),
    )]
    fn test_mul_monomial_proj(
        op_proj: RustNonCommutativeOperator,
        #[case] rhs: RustNonCommutativeMonomial,
        #[case] expected: RustNonCommutativeMonomial,
    ) {
        assert_eq!((&op_proj * &rhs).unwrap(), expected);
        assert_eq!((&op_proj * rhs).unwrap(), expected);
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
                    RustNonCommutativeMonomial::new(
                        vec![
                            op,
                            RustNonCommutativeOperator::new(
                                'x',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            op,
                            RustNonCommutativeOperator::new(
                                'y',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.5, im: 3.5 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new(
                                'x',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    -Complex { re: 1.5, im: 3.5 },
                ),
                (
                    RustNonCommutativeMonomial::from(op),
                    Complex { re: 3.5, im: 5.5 },
                ),
                (
                    RustNonCommutativeMonomial::one(0),
                    Complex::ONE,
                ),
            ]),
        },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            op,
                            op,
                            RustNonCommutativeOperator::new(
                                'x',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            op,
                            op,
                            RustNonCommutativeOperator::new(
                                'y',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.5, im: 3.5 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            op,
                            RustNonCommutativeOperator::new(
                                'x',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    -Complex { re: 1.5, im: 3.5 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![op, op], 0),
                    Complex { re: 3.5, im: 5.5 },
                ),
                (
                    RustNonCommutativeMonomial::from(op),
                    Complex::ONE,
                ),
            ]),
        },
    )]
    fn test_mul_polynomial(
        op: RustNonCommutativeOperator,
        #[case] rhs: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&op * &rhs).unwrap(), expected);
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
                    RustNonCommutativeMonomial::new(
                        vec![
                            op_proj,
                            RustNonCommutativeOperator::new(
                                'x',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.5, im: 3.5 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            op_proj,
                            RustNonCommutativeOperator::new(
                                'y',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new(
                                'x',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    -Complex { re: 1.5, im: 3.5 },
                ),
                (
                    RustNonCommutativeMonomial::from(op_proj),
                    Complex { re: 3.5, im: 5.5 },
                ),
                (
                    RustNonCommutativeMonomial::one(0),
                    Complex::ONE,
                ),
            ]),
        },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            op_proj,
                            RustNonCommutativeOperator::new(
                                'y',
                                1,
                                false,
                                false,
                                false,
                                0
                            ),
                        ],
                        0
                    ),
                    Complex { re: 1.2, im: 3.4 },
                ),
                (
                    RustNonCommutativeMonomial::from(op_proj),
                    Complex { re: 4.5, im: 5.5 },
                ),
            ]),
        },
    )]
    fn test_mul_polynomial_proj(
        op_proj: RustNonCommutativeOperator,
        #[case] rhs: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&op_proj * &rhs).unwrap(), expected);
    }

    #[rstest]
    fn test_pow(op: RustNonCommutativeOperator) {
        assert_eq!(op.pow(0).unwrap(), RustNonCommutativeMonomial::one(0));
        assert_eq!(op.pow(1).unwrap(), RustNonCommutativeMonomial::new(vec![op], 0));
        assert_eq!(op.pow(2).unwrap(), RustNonCommutativeMonomial::new(vec![op, op], 0));
        assert_eq!(op.pow(5).unwrap(), RustNonCommutativeMonomial::new(vec![op, op, op, op, op], 0));
    }

    #[rstest]
    fn test_pow_proj(op_proj: RustNonCommutativeOperator) {
        assert_eq!(op_proj.pow(0).unwrap(), RustNonCommutativeMonomial::one(0));
        assert_eq!(op_proj.pow(1).unwrap(), RustNonCommutativeMonomial::new(vec![op_proj], 0));
        assert_eq!(op_proj.pow(2).unwrap(), RustNonCommutativeMonomial::new(vec![op_proj], 0));
        assert_eq!(op_proj.pow(5).unwrap(), RustNonCommutativeMonomial::new(vec![op_proj], 0));
    }

    #[rstest]
    fn test_mul_operator_different_party(op: RustNonCommutativeOperator) {
        let op_party1 = RustNonCommutativeOperator::new('x', 2, false, false, false, 1);
        assert!((&op * &op_party1).is_err());
    }

    #[rstest]
    fn test_mul_monomial_different_party(op: RustNonCommutativeOperator) {
        let op_party1 = RustNonCommutativeOperator::new('x', 3, false, false, false, 1);
        let mon_party1 = RustNonCommutativeMonomial::from(op_party1);
        assert!((&op * &mon_party1).is_err());
    }
}
