use std::collections::BTreeMap;
use std::ops::{Add, Mul, Sub};

use num_complex::Complex;
use num_traits::Pow;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyNone};

use crate::polynomials::monomial::{AdjointTrait, OneWithMomentMatrixId};
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::{
    PythonNonCommutativeMonomial, RustNonCommutativeMonomial,
};
use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::PythonNonCommutativeOperator;
use crate::polynomials::polynomial::{Polynomial, PolynomialTrait};
use crate::relaxations::constraint::{ConstraintKind, make_noncommutative_constraint};

pub(crate) type RustNonCommutativePolynomial<Scalar> = Polynomial<RustNonCommutativeMonomial, Scalar>;
pub(crate) type RustRealCoefficientsNonCommutativePolynomial = RustNonCommutativePolynomial<f64>;

pub(crate) type RustComplexCoefficientsNonCommutativePolynomial = RustNonCommutativePolynomial<Complex<f64>>;

/// A polynomial in non-commutative operators with real coefficients.
///
/// Instances are typically obtained as the result of arithmetic on
/// [`NonCommutativeOperator`] or [`NonCommutativeMonomial`] objects.
///
/// # Arithmetic
/// Mixed arithmetic with a complex scalar or a
/// [`ComplexCoefficientsNonCommutativePolynomial`] automatically promotes the
/// result to a [`ComplexCoefficientsNonCommutativePolynomial`].
#[pyclass(
    frozen,
    module = "ncpoleon.polynomials.commutative_polynomials",
    name = "RealCoefficientsNonCommutativePolynomial"
)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PythonRealCoefficientsNonCommutativePolynomial(
    pub(crate) RustRealCoefficientsNonCommutativePolynomial,
);

/// A polynomial in non-commutative operators with complex coefficients.
///
/// Instances are typically obtained as the result of arithmetic on
/// [`NonCommutativeOperator`] or [`NonCommutativeMonomial`] objects with
/// complex additive or multiplicative factors.
// FIXME: Seems like we can remove the module part, maybe it's because we add to the module
// manually? If yes, remove
#[pyclass(
    frozen,
    module = "ncpoleon.polynomials.commutative_polynomials",
    name = "ComplexCoefficientsNonCommutativePolynomial"
)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PythonComplexCoefficientsNonCommutativePolynomial(
    pub(crate) RustComplexCoefficientsNonCommutativePolynomial,
);

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonRealCoefficientsNonCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(poly) = value.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Ok(poly.get().clone())
        } else if let Ok(mon) = value.cast::<PythonNonCommutativeMonomial>() {
            Ok(Self(RustRealCoefficientsNonCommutativePolynomial {
                data: BTreeMap::from([(mon.get().0.clone(), 1.0)]),
            }))
        } else if let Ok(op) = value.cast::<PythonNonCommutativeOperator>() {
            Ok(Self(RustRealCoefficientsNonCommutativePolynomial {
                data: BTreeMap::from([(RustNonCommutativeMonomial::from(op.get().0), 1.0)]),
            }))
        } else if let Ok(f64_value) = value.extract::<f64>() {
            Ok(Self(RustRealCoefficientsNonCommutativePolynomial {
                // Caution! We can't know the moment matrix index when converting here, so we
                // instead set it to zero, and it is then the responsibility of the caller to
                // sanitize the moment_matrix_id
                data: BTreeMap::from([(RustNonCommutativeMonomial::one(0), f64_value)]),
            }))
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonRealCoefficientsNonCommutativePolynomial"))
        }
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for PythonRealCoefficientsNonCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonComplexCoefficientsNonCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(poly) = value.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            Ok(poly.get().clone())
        } else if let Ok(complex_value) = value.extract::<Complex<f64>>() {
            Ok(Self(RustComplexCoefficientsNonCommutativePolynomial {
                // Caution! We can't know the moment matrix index when converting here, so we
                // instead set it to zero, and it is then the responsibility of the caller to
                // sanitize the moment_matrix_id
                data: BTreeMap::from([(RustNonCommutativeMonomial::one(0), complex_value)]),
            }))
        } else if let Ok(real_poly) = PythonRealCoefficientsNonCommutativePolynomial::try_from(value) {
            Ok(real_poly.into())
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonComplexCoefficientsNonCommutativePolynomial"))
        }
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for PythonComplexCoefficientsNonCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl From<&RustRealCoefficientsNonCommutativePolynomial> for RustComplexCoefficientsNonCommutativePolynomial {
    fn from(value: &RustRealCoefficientsNonCommutativePolynomial) -> Self {
        Self { data: value.data.iter().map(|(mon, &coeff)| (mon.clone(), Complex::from(coeff))).collect() }
    }
}

impl From<RustRealCoefficientsNonCommutativePolynomial> for RustComplexCoefficientsNonCommutativePolynomial {
    fn from(value: RustRealCoefficientsNonCommutativePolynomial) -> Self {
        Self { data: value.data.into_iter().map(|(mon, coeff)| (mon, Complex::from(coeff))).collect() }
    }
}

impl From<PythonRealCoefficientsNonCommutativePolynomial> for PythonComplexCoefficientsNonCommutativePolynomial {
    fn from(value: PythonRealCoefficientsNonCommutativePolynomial) -> Self {
        Self(value.0.into())
    }
}

impl PolynomialTrait for RustRealCoefficientsNonCommutativePolynomial {
    fn chop(&self, delta: f64) -> Self {
        let mut res = BTreeMap::new();
        for (monomial, &coeff) in self.data.iter() {
            if coeff.abs() > delta {
                res.insert(monomial.clone(), coeff);
            }
        }
        Self { data: res }
    }

    fn degree(&self) -> u8 {
        self.data.keys().map(|monomial| monomial.len()).max().unwrap_or_default()
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl AdjointTrait for RustRealCoefficientsNonCommutativePolynomial {
    fn adjoint(&self) -> Self {
        Self { data: self.data.iter().map(|(monomial, &coeff)| (monomial.adjoint(), coeff)).collect() }
    }
}

impl PolynomialTrait for RustComplexCoefficientsNonCommutativePolynomial {
    fn chop(&self, delta: f64) -> Self {
        let mut res = BTreeMap::new();
        for (monomial, &coeff) in self.data.iter() {
            let mut new_coeff = coeff;
            if new_coeff.re.abs() <= delta {
                new_coeff.re = 0f64;
            }
            if new_coeff.im.abs() <= delta {
                new_coeff.im = 0f64;
            }
            if new_coeff.norm() > delta {
                res.insert(monomial.clone(), new_coeff);
            }
        }
        Self { data: res }
    }

    fn degree(&self) -> u8 {
        self.data.keys().map(|monomial| monomial.len()).max().unwrap_or_default()
    }

    fn is_real(&self) -> bool {
        false
    }
}

impl AdjointTrait for RustComplexCoefficientsNonCommutativePolynomial {
    fn adjoint(&self) -> Self {
        Self { data: self.data.iter().map(|(monomial, coeff)| (monomial.adjoint(), coeff.conj())).collect() }
    }
}

impl Add<Complex<f64>> for &RustRealCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn add(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) + rhs
    }
}

impl Add<f64> for &RustComplexCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn add(self, rhs: f64) -> Self::Output {
        self + Complex::from(rhs)
    }
}

impl Add<Complex<f64>> for RustRealCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn add(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) + rhs
    }
}

impl Sub<Complex<f64>> for &RustRealCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn sub(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) - rhs
    }
}

impl Sub<f64> for &RustComplexCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Complex::from(rhs)
    }
}

impl Sub<Complex<f64>> for RustRealCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn sub(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) - rhs
    }
}

impl Mul<Complex<f64>> for &RustRealCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) * rhs
    }
}

impl Mul<f64> for &RustComplexCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Complex::from(rhs)
    }
}

impl Mul<Complex<f64>> for RustRealCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) * rhs
    }
}

impl Add<&RustComplexCoefficientsNonCommutativePolynomial> for &RustRealCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn add(self, rhs: &RustComplexCoefficientsNonCommutativePolynomial) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) + rhs
    }
}

impl Add<&RustRealCoefficientsNonCommutativePolynomial> for &RustComplexCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn add(self, rhs: &RustRealCoefficientsNonCommutativePolynomial) -> Self::Output {
        self + RustComplexCoefficientsNonCommutativePolynomial::from(rhs)
    }
}

impl Sub<&RustComplexCoefficientsNonCommutativePolynomial> for &RustRealCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn sub(self, rhs: &RustComplexCoefficientsNonCommutativePolynomial) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) - rhs
    }
}

impl Sub<&RustRealCoefficientsNonCommutativePolynomial> for &RustComplexCoefficientsNonCommutativePolynomial {
    type Output = RustComplexCoefficientsNonCommutativePolynomial;

    fn sub(self, rhs: &RustRealCoefficientsNonCommutativePolynomial) -> Self::Output {
        self - RustComplexCoefficientsNonCommutativePolynomial::from(rhs)
    }
}

impl Mul<&RustComplexCoefficientsNonCommutativePolynomial> for &RustRealCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn mul(self, rhs: &RustComplexCoefficientsNonCommutativePolynomial) -> Self::Output {
        RustComplexCoefficientsNonCommutativePolynomial::from(self) * rhs
    }
}

impl Mul<&RustRealCoefficientsNonCommutativePolynomial> for &RustComplexCoefficientsNonCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsNonCommutativePolynomial, String>;

    fn mul(self, rhs: &RustRealCoefficientsNonCommutativePolynomial) -> Self::Output {
        self * RustComplexCoefficientsNonCommutativePolynomial::from(rhs)
    }
}

#[pymethods]
impl PythonRealCoefficientsNonCommutativePolynomial {
    /// Evaluate the polynomial under a monomial-to-value mapping.
    ///
    /// Each [`NonCommutativeMonomial`] in the polynomial is looked up in
    /// `mapping`; its image is multiplied by the corresponding coefficient
    /// and all terms are summed.  The values in `mapping` can be any Python
    /// objects that support `*` and `+` via the `__add__` and `__mul__`
    /// dunder methods. In complex-valued problems, complex variables must also
    /// have a `conj` method.
    ///
    /// # Errors
    /// Raises `ValueError` if any monomial key is missing from `mapping`, or
    /// if `self` is the zero polynomial.
    pub(crate) fn change_variables<'py>(&self, mapping: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let res = self
            .0
            .data
            .iter()
            .map(|(mon, &coeff)| {
                let mapped = mapping.get_item(PythonNonCommutativeMonomial(mon.clone()));

                if let Ok(Some(mapped)) = mapped {
                    mapped.mul(coeff)
                } else {
                    Err(PyValueError::new_err(format!(
                        "Couldn't find monomial {} in the provided mapping.",
                        mon.__str__()
                    )))
                }
            })
            .reduce(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left.add(right),
                (Ok(_), Err(err_right)) => Err(err_right),
                (Err(err_left), Ok(_)) => Err(err_left),
                (Err(err_left), Err(_)) => Err(err_left),
            });
        if let Some(res) = res { res } else { Err(PyValueError::new_err("Can't replace the Zero polynomial.")) }
    }

    fn as_dict(&self) -> BTreeMap<PythonNonCommutativeMonomial, f64> {
        self.0
            .data
            .iter()
            .map(|(rust_monomial, &coeff)| (PythonNonCommutativeMonomial(rust_monomial.clone()), coeff))
            .collect()
    }

    fn by_moment_matrix_id(&self) -> BTreeMap<u8, Self> {
        self.0.by_moment_matrix_id().into_iter().map(|(mm_id, poly)| (mm_id, Self(poly))).collect()
    }

    #[getter]
    fn is_real(&self) -> bool {
        true
    }

    pub(crate) fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            Self(&self.0 + &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            Self(&self.0 + &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Self(&self.0 + &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 + &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            Self((&self.0 + lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
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
            Self((&self.0 + lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
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
            Self(&self.0 - &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            Self(&self.0 - &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Self(&self.0 - &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 - &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            Self((&self.0 - lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
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
            Self((-&self.0 + lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
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
            Self((&self.0 * &op.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            Self((&self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Self((&self.0 * &poly_real.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            PythonComplexCoefficientsNonCommutativePolynomial(
                (&RustComplexCoefficientsNonCommutativePolynomial::from(&self.0) * &poly_complex.get().0)
                    .map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            Self(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            Self(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            Self(&self.0 * (1.0 / lambda_real)).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            PythonComplexCoefficientsNonCommutativePolynomial(&self.0 * (1.0 / lambda_complex)).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __neg__(&self) -> Self {
        Self(-&self.0)
    }

    fn __pow__<'py>(&self, power: u8, _modulo: &Bound<'py, PyNone>) -> PyResult<Self> {
        Ok(Self((&self.0).pow(power).map_err(PyValueError::new_err)?))
    }

    /// Return the degree of the polynomial, which is the maximum total degree among all its
    /// monomials.
    pub(crate) fn degree(&self) -> u8 {
        self.0.degree()
    }

    /// Return the adjoint (Hermitian conjugate) of this polynomial.
    fn adjoint(&self) -> Self {
        Self(self.0.adjoint())
    }

    /// Check if this polynomial is nil. Additionally chop it using its chop method beforehand if
    /// the delta parameter is specified
    #[pyo3(signature=(delta=None))]
    fn is_zero(&self, delta: Option<f64>) -> bool {
        if let Some(delta) = delta { self.0.chop(delta).data.is_empty() } else { self.0.data.is_empty() }
    }

    /// Return a polynomial identical to the one this method has been called with, at the exception
    /// that coefficients whose absolute value is below `delta` are removed.
    #[pyo3(signature=(delta=1e-10))]
    fn chop(&self, delta: f64) -> Self {
        Self(self.0.chop(delta))
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
}

#[pymethods]
impl PythonComplexCoefficientsNonCommutativePolynomial {
    /// Evaluate the polynomial under a monomial-to-value mapping.
    ///
    /// See [`RealCoefficientsNonCommutativePolynomial.change_variables`] for
    /// full documentation.
    pub(crate) fn change_variables<'py>(&self, mapping: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let res = self
            .0
            .data
            .iter()
            .map(|(mon, &coeff)| {
                let mapped = mapping.get_item(PythonNonCommutativeMonomial(mon.clone()));

                if let Ok(Some(mapped)) = mapped {
                    mapped.mul(coeff)
                } else {
                    Err(PyValueError::new_err(format!(
                        "Couldn't find monomial {} in the provided mapping.",
                        mon.__str__()
                    )))
                }
            })
            .reduce(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left.add(right),
                (Ok(_), Err(err_right)) => Err(err_right),
                (Err(err_left), Ok(_)) => Err(err_left),
                (Err(err_left), Err(_)) => Err(err_left),
            });
        if let Some(res) = res { res } else { Err(PyValueError::new_err("Can't replace the Zero polynomial.")) }
    }

    fn as_dict(&self) -> BTreeMap<PythonNonCommutativeMonomial, Complex<f64>> {
        self.0
            .data
            .iter()
            .map(|(rust_monomial, &coeff)| (PythonNonCommutativeMonomial(rust_monomial.clone()), coeff))
            .collect()
    }

    fn by_moment_matrix_id(&self) -> BTreeMap<u8, Self> {
        self.0.by_moment_matrix_id().into_iter().map(|(mm_id, poly)| (mm_id, Self(poly))).collect()
    }

    #[getter]
    fn is_real(&self) -> bool {
        false
    }

    pub(crate) fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            Self(&self.0 + &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            Self(&self.0 + &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Self(&self.0 + &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            Self(&self.0 + &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            Self((&self.0 + lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self((&self.0 + lambda_complex).map_err(PyValueError::new_err)?).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            Self((&self.0 + lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self((&self.0 + lambda_complex).map_err(PyValueError::new_err)?).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __sub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            Self(&self.0 - &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            Self(&self.0 - &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Self(&self.0 - &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            Self(&self.0 - &poly_complex.get().0).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            Self((&self.0 - lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self((&self.0 - lambda_complex).map_err(PyValueError::new_err)?).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rsub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            let neg = -&self.0;
            Self((&neg + lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self((-&self.0 + lambda_complex).map_err(PyValueError::new_err)?).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __mul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonNonCommutativeOperator>() {
            Self((&self.0 * &op.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonNonCommutativeMonomial>() {
            Self((&self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
            Self(
                (&self.0 * &RustComplexCoefficientsNonCommutativePolynomial::from(&poly_real.get().0))
                    .map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
            Self((&self.0 * &poly_complex.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(lambda_real) = other.extract::<f64>() {
            Self(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            Self(&self.0 * lambda_real).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self(&self.0 * lambda_complex).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(lambda_real) = other.extract::<f64>() {
            Self(&self.0 * (1.0 / lambda_real)).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self(&self.0 * (1.0 / lambda_complex)).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __neg__(&self) -> Self {
        Self(-&self.0)
    }

    fn __pow__<'py>(&self, power: u8, _modulo: &Bound<'py, PyNone>) -> PyResult<Self> {
        Ok(Self((&self.0).pow(power).map_err(PyValueError::new_err)?))
    }

    /// Return the degree of the polynomial, which is the maximum total degree among all its
    /// monomials.
    pub(crate) fn degree(&self) -> u8 {
        self.0.degree()
    }

    /// Return the adjoint (Hermitian conjugate) of this polynomial.
    fn adjoint(&self) -> Self {
        Self(self.0.adjoint())
    }

    /// Return a polynomial identical to the one this method has been called with, at the exception
    /// that:
    ///  - coefficients that have their real part lower in absolute value than delta having their
    /// real part set to zero;
    ///  - coefficients that have their imaginary part lower in absolute value than delta having
    /// their imaginary part set to zero;
    /// - coefficients that have their modulus lower than delta (after the potential aforementioned
    /// modifications on their real and imaginary parts) being removed.
    #[pyo3(signature=(delta=1e-10))]
    pub(crate) fn chop(&self, delta: f64) -> Self {
        Self(self.0.chop(delta))
    }

    /// Check if this polynomial is nil. Additionally chop it using its chop method beforehand if
    /// the delta parameter is specified
    #[pyo3(signature=(delta=None))]
    fn is_zero(&self, delta: Option<f64>) -> bool {
        if let Some(delta) = delta { self.0.chop(delta).data.is_empty() } else { self.0.data.is_empty() }
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
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use num_complex::Complex;
    use num_traits::{Pow, Zero};
    use rstest::{fixture, rstest};

    use super::*;
    use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::RustNonCommutativeMonomial;
    use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::RustNonCommutativeOperator;

    #[fixture]
    fn real_poly() -> RustRealCoefficientsNonCommutativePolynomial {
        RustRealCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), 1.0),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    2.0,
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    -1.0,
                ),
            ]),
        }
    }

    #[fixture]
    fn complex_poly() -> RustComplexCoefficientsNonCommutativePolynomial {
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex::I,
                ),
            ]),
        }
    }

    #[rstest]
    fn test_adjoint_real(real_poly: RustRealCoefficientsNonCommutativePolynomial) {
        let adjoint = real_poly.adjoint();
        assert_eq!(
            adjoint,
            RustRealCoefficientsNonCommutativePolynomial {
                data: BTreeMap::from([
                    (RustNonCommutativeMonomial::one(0), 1.0,),
                    (
                        RustNonCommutativeMonomial::new(
                            vec![
                                RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                                RustNonCommutativeOperator::new('x', 0, true, false, false, 0),
                            ],
                            0
                        ),
                        2.0,
                    ),
                    (
                        RustNonCommutativeMonomial::new(
                            vec![
                                RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                                RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            ],
                            0
                        ),
                        -1.0,
                    ),
                ]),
            }
        );
    }

    #[rstest]
    fn test_adjoint_complex(complex_poly: RustComplexCoefficientsNonCommutativePolynomial) {
        let adjoint = complex_poly.adjoint();
        assert_eq!(
            adjoint,
            RustComplexCoefficientsNonCommutativePolynomial {
                data: BTreeMap::from([
                    (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: -2.0 },),
                    (
                        RustNonCommutativeMonomial::new(
                            vec![
                                RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                                RustNonCommutativeOperator::new('x', 0, true, false, false, 0),
                            ],
                            0
                        ),
                        Complex::ONE,
                    ),
                    (
                        RustNonCommutativeMonomial::new(
                            vec![
                                RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                                RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            ],
                            0
                        ),
                        -Complex::I,
                    ),
                ]),
            }
        );
    }

    #[rstest]
    #[case(Complex::ZERO, RustComplexCoefficientsNonCommutativePolynomial::zero())]
    #[case(
        Complex { re: 1.0, im: 2.0 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: -3.0, im: 4.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex { re: -2.0, im: 1.0 },
                ),
            ]),
        }
    )]
    fn test_mul_complex(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly * rhs, expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_mul_operator(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustNonCommutativeOperator,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&complex_poly * &rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeMonomial::one(0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
        ], 0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_mul_monomial(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustNonCommutativeMonomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&complex_poly * &rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial::zero(),
        RustComplexCoefficientsNonCommutativePolynomial::zero()
    )]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
            ]),
        },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: -3.0, im: 4.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex { re: -2.0, im: 1.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_mul_polynomial(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&complex_poly * &rhs).unwrap(), expected);
        assert_eq!((&complex_poly * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex { re: 0.0, im: 0.0 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: 1.5, im: 3.5 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 2.5, im: 5.5 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: -1.0, im: -2.0 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_complex(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&complex_poly + rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex { re: 0.0, im: 0.0 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: 1.5, im: 3.5 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: -0.5, im: -1.5 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: 1.0, im: 2.0 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_sub_complex(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&complex_poly - rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeOperator::new('z', 0, false, false, false, 0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('z', 0, false, false, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
            ]),
        }
    )]
    fn test_add_operator(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustNonCommutativeOperator,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly + &rhs, expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeOperator::new('z', 0, false, false, false, 0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('z', 0, false, false, false, 0),
                    ], 0),
                    -Complex::ONE,
                ),
            ]),
        }
    )]
    fn test_sub_operator(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustNonCommutativeOperator,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly - &rhs, expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeMonomial::one(0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 2.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
        ], 0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_monomial(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustNonCommutativeMonomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly + &rhs, expected);
    }

    #[rstest]
    #[case(
        RustNonCommutativeMonomial::one(0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustNonCommutativeMonomial::new(vec![
            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
        ], 0),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_sub_monomial(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustNonCommutativeMonomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly - &rhs, expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial::zero(),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    -Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 2.0, im: 4.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_polynomial(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly + &rhs, expected);
        assert_eq!(&complex_poly + rhs, expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial::zero(),
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    -Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    Complex::I,
                ),
            ]),
        },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex::I,
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                    ], 0),
                    -Complex::I,
                ),
            ]),
        }
    )]
    fn test_sub_polynomial(
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] rhs: RustComplexCoefficientsNonCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!(&complex_poly - &rhs, expected);
        assert_eq!(&complex_poly - rhs, expected);
    }

    #[rstest]
    fn test_neg(complex_poly: RustComplexCoefficientsNonCommutativePolynomial) {
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: -1.0, im: -2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    -Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    -Complex::I,
                ),
            ]),
        };
        assert_eq!(-&complex_poly, expected);
        assert_eq!(-complex_poly, expected);
    }

    #[test]
    fn test_pow() {
        let poly = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex::ONE),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 0, false, false, false, 0)],
                        0,
                    ),
                    Complex::I,
                ),
            ]),
        };
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([(RustNonCommutativeMonomial::one(0), Complex::ONE)]),
        };
        assert_eq!((&poly).pow(0u8).unwrap(), expected);
        assert_eq!((&poly).pow(1u8).unwrap(), poly.clone());

        // (1 + i*x0)^2 = 1 + 2i*x0 - [x0,x0]
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex::ONE),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 0, false, false, false, 0)],
                        0,
                    ),
                    Complex { re: 0.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: -1.0, im: 0.0 },
                ),
            ]),
        };
        assert_eq!((&poly).pow(2u8).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex { re: 1.5, im: 2.5 },
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 2.5, im: 2.5 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
            ]),
        }
    )]
    #[case(
        Complex::ZERO,
        RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 1.0, im: 0.0 }),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                    ], 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(vec![
                        RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                        RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                    ], 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
            ]),
        }
    )]
    fn test_real_poly_add_complex_scalar(
        real_poly: RustRealCoefficientsNonCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        assert_eq!((&real_poly + rhs).unwrap(), expected);
        assert_eq!((real_poly + rhs).unwrap(), expected);
    }

    #[rstest]
    fn test_real_poly_sub_complex_scalar(real_poly: RustRealCoefficientsNonCommutativePolynomial) {
        let rhs = Complex { re: 0.5, im: 1.5 };
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.5, im: -1.5 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex { re: -1.0, im: 0.0 },
                ),
            ]),
        };
        assert_eq!((&real_poly - rhs).unwrap(), expected);
        assert_eq!((real_poly - rhs).unwrap(), expected);
    }

    #[rstest]
    fn test_real_poly_mul_complex_scalar(real_poly: RustRealCoefficientsNonCommutativePolynomial) {
        let rhs = Complex::I;
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex::I),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 0.0, im: 2.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex { re: 0.0, im: -1.0 },
                ),
            ]),
        };
        assert_eq!(&real_poly * rhs, expected);
        assert_eq!(real_poly * rhs, expected);
    }

    #[rstest]
    fn test_complex_poly_add_f64(complex_poly: RustComplexCoefficientsNonCommutativePolynomial) {
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 4.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex::I,
                ),
            ]),
        };
        assert_eq!((&complex_poly + 3.0).unwrap(), expected);
    }

    #[rstest]
    fn test_complex_poly_sub_f64(complex_poly: RustComplexCoefficientsNonCommutativePolynomial) {
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex::ONE,
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex::I,
                ),
            ]),
        };
        assert_eq!((&complex_poly - 1.0).unwrap(), expected);
    }

    #[rstest]
    fn test_complex_poly_mul_f64(complex_poly: RustComplexCoefficientsNonCommutativePolynomial) {
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 2.0, im: 4.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex { re: 0.0, im: 2.0 },
                ),
            ]),
        };
        assert_eq!(&complex_poly * 2.0, expected);
    }

    #[rstest]
    fn test_real_poly_add_complex_poly(
        real_poly: RustRealCoefficientsNonCommutativePolynomial,
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 2.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 3.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex { re: -1.0, im: 1.0 },
                ),
            ]),
        };
        assert_eq!(&real_poly + &complex_poly, expected);
        assert_eq!(&complex_poly + &real_poly, expected);
    }

    #[rstest]
    fn test_real_poly_sub_complex_poly(
        real_poly: RustRealCoefficientsNonCommutativePolynomial,
        complex_poly: RustComplexCoefficientsNonCommutativePolynomial,
    ) {
        let real_minus_complex = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.0, im: -2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 1.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex { re: -1.0, im: -1.0 },
                ),
            ]),
        };
        assert_eq!(&real_poly - &complex_poly, real_minus_complex);

        let complex_minus_real = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, true, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: -1.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('y', 0, false, true, true, 0),
                            RustNonCommutativeOperator::new('y', 1, false, true, true, 0),
                        ],
                        0,
                    ),
                    Complex { re: 1.0, im: 1.0 },
                ),
            ]),
        };
        assert_eq!(&complex_poly - &real_poly, complex_minus_real);
    }

    #[rstest]
    fn test_real_poly_mul_complex_poly() {
        // Use simpler polynomials to keep the test readable
        let real = RustRealCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), 2.0),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 0, false, false, false, 0)],
                        0,
                    ),
                    3.0,
                ),
            ]),
        };
        let complex = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex::I),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 1, false, false, false, 0)],
                        0,
                    ),
                    Complex::ONE,
                ),
            ]),
        };

        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 0, false, false, false, 0)],
                        0,
                    ),
                    Complex { re: 0.0, im: 3.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 1, false, false, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 3.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 1, false, false, false, 0)],
                        0,
                    ),
                    Complex { re: 2.0, im: 0.0 },
                ),
            ]),
        };
        assert_eq!((&real * &complex).unwrap(), expected);

        let expected_rev = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([
                (RustNonCommutativeMonomial::one(0), Complex { re: 0.0, im: 2.0 }),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 0, false, false, false, 0)],
                        0,
                    ),
                    Complex { re: 0.0, im: 3.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![RustNonCommutativeOperator::new('x', 1, false, false, false, 0)],
                        0,
                    ),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustNonCommutativeMonomial::new(
                        vec![
                            RustNonCommutativeOperator::new('x', 1, false, false, false, 0),
                            RustNonCommutativeOperator::new('x', 0, false, false, false, 0),
                        ],
                        0,
                    ),
                    Complex { re: 3.0, im: 0.0 },
                ),
            ]),
        };
        assert_eq!((&complex * &real).unwrap(), expected_rev);
    }
}
