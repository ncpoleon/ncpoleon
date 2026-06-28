use std::collections::BTreeMap;
use std::ops::{Add, Mul, Sub};

use num_complex::Complex;
use num_traits::Pow;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyNone};

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::{
    PythonCommutativeMonomial, RustCommutativeMonomial,
};
use crate::polynomials::commutative_polynomials::operators::commutative_operator::PythonCommutativeOperator;
use crate::polynomials::monomial::{AdjointTrait, OneWithMomentMatrixId};
use crate::polynomials::polynomial::{Polynomial, PolynomialTrait};
use crate::relaxations::constraint::{ConstraintKind, make_commutative_constraint};

pub(crate) type RustCommutativePolynomial<Scalar> = Polynomial<RustCommutativeMonomial, Scalar>;
pub(crate) type RustRealCoefficientsCommutativePolynomial = RustCommutativePolynomial<f64>;
pub(crate) type RustComplexCoefficientsCommutativePolynomial = RustCommutativePolynomial<Complex<f64>>;

/// A polynomial in commutative variables with real coefficients.
///
/// Instances are typically obtained as the result of arithmetic on
/// [`CommutativeOperator`] or [`CommutativeMonomial`] objects.
///
/// # Arithmetic
/// Mixed arithmetic with a complex scalar or a
/// [`ComplexCoefficientsCommutativePolynomial`] automatically promotes the
/// result to a [`ComplexCoefficientsCommutativePolynomial`].
#[pyclass(
    frozen,
    module = "ncpoleon.polynomials.commutative_polynomials",
    name = "RealCoefficientsCommutativePolynomial"
)]
#[derive(Clone)]
// TODO: Seems like we can remove this, maybe it's because we add to the module manually? If yes,
// remove
pub(crate) struct PythonRealCoefficientsCommutativePolynomial(pub(crate) RustRealCoefficientsCommutativePolynomial);

/// A polynomial in commutative variables with complex  coefficients.
///
/// Instances are typically obtained as the result of arithmetic on
/// [`CommutativeOperator`] or [`CommutativeMonomial`] objects with
/// complex additive or multiplicative factors.
#[pyclass(
    frozen,
    module = "ncpoleon.polynomials.commutative_polynomials",
    name = "ComplexCoefficientsCommutativePolynomial"
)]
#[derive(Clone)]
// TODO: Seems like we can remove this, maybe it's because we add to the module manually? If yes,
// remove
pub(crate) struct PythonComplexCoefficientsCommutativePolynomial(
    pub(crate) RustComplexCoefficientsCommutativePolynomial,
);

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonRealCoefficientsCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(poly) = value.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            Ok(poly.get().clone())
        } else if let Ok(mon) = value.cast::<PythonCommutativeMonomial>() {
            Ok(mon.get().clone().into())
        } else if let Ok(op) = value.cast::<PythonCommutativeOperator>() {
            Ok((*op.get()).into())
        } else if let Ok(f64_value) = value.extract::<f64>() {
            // Caution! We can't know the moment matrix index when converting here, so we instead set it to zero, and
            // it is then the responsability of the caller to sanitize the moment_matrix_id
            Ok(Self(RustRealCoefficientsCommutativePolynomial {
                data: BTreeMap::from([(RustCommutativeMonomial::one(0), f64_value)]),
            }))
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonRealCoefficientsCommutativePolynomial"))
        }
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for PythonRealCoefficientsCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonComplexCoefficientsCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(poly) = value.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
            Ok(poly.get().clone())
        } else if let Ok(complex_value) = value.extract::<Complex<f64>>() {
            // Caution! We can't know the moment matrix index when converting here, so we instead set it to zero, and
            // it is then the responsability of the caller to sanitize the moment_matrix_id
            Ok(Self(RustComplexCoefficientsCommutativePolynomial {
                data: BTreeMap::from([(RustCommutativeMonomial::one(0), complex_value)]),
            }))
        } else if let Ok(real_poly) = PythonRealCoefficientsCommutativePolynomial::try_from(value) {
            Ok(real_poly.into())
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonComplexCoefficientsCommutativePolynomial"))
        }
    }
}

impl<'py> TryFrom<Bound<'py, PyAny>> for PythonComplexCoefficientsCommutativePolynomial {
    type Error = PyErr;

    fn try_from(value: Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl From<PythonCommutativeMonomial> for PythonRealCoefficientsCommutativePolynomial {
    fn from(value: PythonCommutativeMonomial) -> Self {
        Self(value.0.into())
    }
}

impl From<PythonCommutativeOperator> for PythonRealCoefficientsCommutativePolynomial {
    fn from(value: PythonCommutativeOperator) -> Self {
        Self(value.0.into())
    }
}

impl From<PythonRealCoefficientsCommutativePolynomial> for PythonComplexCoefficientsCommutativePolynomial {
    fn from(value: PythonRealCoefficientsCommutativePolynomial) -> Self {
        Self(value.0.into())
    }
}

#[pymethods]
impl PythonRealCoefficientsCommutativePolynomial {
    /// Evaluate the polynomial under a monomial-to-value mapping.
    ///
    /// Each [`CommutativeMonomial`] in the polynomial is looked up in
    /// `mapping`; its image is multiplied by the corresponding coefficient
    /// and all terms are summed.  The values in `mapping` can be any Python
    /// objects that support `*` and `+` via the `__add__` and `__mul__`
    /// dunder methods.
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
                let mapped = mapping.get_item(PythonCommutativeMonomial(mon.clone()));

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

    fn as_dict(&self) -> BTreeMap<PythonCommutativeMonomial, f64> {
        self.0
            .data
            .iter()
            .map(|(rust_monomial, &coeff)| (PythonCommutativeMonomial(rust_monomial.clone()), coeff))
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
            PythonRealCoefficientsCommutativePolynomial((&self.0 * &op.get().0).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 * &mon.get().0).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            PythonRealCoefficientsCommutativePolynomial((&self.0 * &poly_real.get().0).map_err(PyValueError::new_err)?)
                .into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
            PythonComplexCoefficientsCommutativePolynomial(
                (RustComplexCoefficientsCommutativePolynomial::from(&self.0) * &poly_complex.get().0)
                    .map_err(PyValueError::new_err)?,
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

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __neg__(&self) -> Self {
        Self(-&self.0)
    }

    fn __pow__<'py>(&self, power: u8, modulo: &Bound<'py, PyNone>) -> PyResult<Py<PyAny>> {
        Self((&self.0).pow(power).map_err(PyValueError::new_err)?).into_py_any(modulo.py())
    }

    /// Return the degree of the polynomial, which is the maximum total degree among all its
    /// monomials.
    pub(crate) fn degree(&self) -> u8 {
        self.0.degree()
    }

    /// Return the complex conjugate of this polynomial.
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
}

#[pymethods]
impl PythonComplexCoefficientsCommutativePolynomial {
    /// Evaluate the polynomial under a monomial-to-value mapping.
    ///
    /// See [`RealCoefficientsCommutativePolynomial.change_variables`] for full
    /// documentation.
    pub(crate) fn change_variables<'py>(&self, mapping: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let res = self
            .0
            .data
            .iter()
            .map(|(mon, &coeff)| {
                let mapped = mapping.get_item(PythonCommutativeMonomial(mon.clone()));

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

    fn as_dict(&self) -> BTreeMap<PythonCommutativeMonomial, Complex<f64>> {
        self.0
            .data
            .iter()
            .map(|(rust_monomial, &coeff)| (PythonCommutativeMonomial(rust_monomial.clone()), coeff))
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
        if let Ok(op) = other.cast::<PythonCommutativeOperator>() {
            Self(&self.0 + &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            Self(&self.0 + &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            Self(&self.0 + &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
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
        // __radd__ is only called if __add__ isn't supported on other, which means that it's either
        // a scalar, or not supported
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
        if let Ok(op) = other.cast::<PythonCommutativeOperator>() {
            Self(&self.0 - &op.get().0).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            Self(&self.0 - &mon.get().0).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            Self(&self.0 - &poly_real.get().0).into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
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
        // __rsub__ is only called if __sub__ isn't supported on other, which means that it's either
        // a scalar, or not supported
        if let Ok(lambda_real) = other.extract::<f64>() {
            Self(-(&self.0 - lambda_real).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(lambda_complex) = other.extract::<Complex<f64>>() {
            Self((-&self.0 + lambda_complex).map_err(PyValueError::new_err)?).into_py_any(py)
        } else {
            Ok(py.NotImplemented().into_any())
        }
    }

    fn __mul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(op) = other.cast::<PythonCommutativeOperator>() {
            Self((&self.0 * &op.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(mon) = other.cast::<PythonCommutativeMonomial>() {
            Self((&self.0 * &mon.get().0).map_err(PyValueError::new_err)?).into_py_any(py)
        } else if let Ok(poly_real) = other.cast::<PythonRealCoefficientsCommutativePolynomial>() {
            Self(
                (&self.0 * RustComplexCoefficientsCommutativePolynomial::from(&poly_real.get().0))
                    .map_err(PyValueError::new_err)?,
            )
            .into_py_any(py)
        } else if let Ok(poly_complex) = other.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
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
        // __rmul__ is only called if __mul__ isn't supported on other, which means that it's either
        // a scalar, or not supported
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
        // Can't divide by anything else than a scalar for now
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

    fn __pow__<'py>(&self, power: u8, modulo: &Bound<'py, PyNone>) -> PyResult<Py<PyAny>> {
        Self((&self.0).pow(power).map_err(PyValueError::new_err)?).into_py_any(modulo.py())
    }

    /// Return the degree of the polynomial, which is the maximum total degree among all its
    /// monomials.    
    pub(crate) fn degree(&self) -> u8 {
        self.0.degree()
    }

    /// Return the complex conjugate of this polynomial.
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
}

impl From<&RustRealCoefficientsCommutativePolynomial> for RustComplexCoefficientsCommutativePolynomial {
    fn from(value: &RustRealCoefficientsCommutativePolynomial) -> Self {
        Self { data: value.data.iter().map(|(mon, &power)| (mon.clone(), Complex::from(power))).collect() }
    }
}

impl From<RustRealCoefficientsCommutativePolynomial> for RustComplexCoefficientsCommutativePolynomial {
    fn from(value: RustRealCoefficientsCommutativePolynomial) -> Self {
        Self { data: value.data.into_iter().map(|(mon, power)| (mon, Complex::from(power))).collect() }
    }
}

impl PolynomialTrait for RustRealCoefficientsCommutativePolynomial {
    fn chop(&self, delta: f64) -> Self {
        let mut res = BTreeMap::new();
        for (monomial, &coeff) in self.data.iter() {
            if coeff.abs() > delta {
                res.insert(monomial.clone(), coeff);
            }
        }
        RustRealCoefficientsCommutativePolynomial { data: res }
    }

    fn degree(&self) -> u8 {
        self.data.keys().map(|monomial| monomial.len()).max().unwrap_or_default()
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl AdjointTrait for RustRealCoefficientsCommutativePolynomial {
    fn adjoint(&self) -> Self {
        Self { data: self.data.iter().map(|(monomial, &coeff)| (monomial.adjoint(), coeff)).collect() }
    }
}

impl Add<Complex<f64>> for &RustRealCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn add(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) + rhs
    }
}

impl Add<f64> for &RustComplexCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn add(self, rhs: f64) -> Self::Output {
        self + Complex::from(rhs)
    }
}

impl Add<Complex<f64>> for RustRealCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn add(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) + rhs
    }
}

impl Sub<Complex<f64>> for &RustRealCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn sub(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) - rhs
    }
}

impl Sub<f64> for &RustComplexCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn sub(self, rhs: f64) -> Self::Output {
        self - Complex::from(rhs)
    }
}

impl Sub<Complex<f64>> for RustRealCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn sub(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) - rhs
    }
}

impl Mul<Complex<f64>> for &RustRealCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) * rhs
    }
}

impl Mul<f64> for &RustComplexCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Complex::from(rhs)
    }
}

impl Mul<Complex<f64>> for RustRealCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) * rhs
    }
}

impl Add<&RustComplexCoefficientsCommutativePolynomial> for &RustRealCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn add(self, rhs: &RustComplexCoefficientsCommutativePolynomial) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) + rhs
    }
}

impl Add<&RustRealCoefficientsCommutativePolynomial> for &RustComplexCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn add(self, rhs: &RustRealCoefficientsCommutativePolynomial) -> Self::Output {
        self + RustComplexCoefficientsCommutativePolynomial::from(rhs)
    }
}

impl Sub<&RustComplexCoefficientsCommutativePolynomial> for &RustRealCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn sub(self, rhs: &RustComplexCoefficientsCommutativePolynomial) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) - rhs
    }
}

impl Sub<&RustRealCoefficientsCommutativePolynomial> for &RustComplexCoefficientsCommutativePolynomial {
    type Output = RustComplexCoefficientsCommutativePolynomial;

    fn sub(self, rhs: &RustRealCoefficientsCommutativePolynomial) -> Self::Output {
        self - RustComplexCoefficientsCommutativePolynomial::from(rhs)
    }
}

impl Mul<&RustComplexCoefficientsCommutativePolynomial> for &RustRealCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn mul(self, rhs: &RustComplexCoefficientsCommutativePolynomial) -> Self::Output {
        RustComplexCoefficientsCommutativePolynomial::from(self) * rhs
    }
}

impl Mul<&RustRealCoefficientsCommutativePolynomial> for &RustComplexCoefficientsCommutativePolynomial {
    type Output = Result<RustComplexCoefficientsCommutativePolynomial, String>;

    fn mul(self, rhs: &RustRealCoefficientsCommutativePolynomial) -> Self::Output {
        self * RustComplexCoefficientsCommutativePolynomial::from(rhs)
    }
}

impl PolynomialTrait for RustComplexCoefficientsCommutativePolynomial {
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
        RustComplexCoefficientsCommutativePolynomial { data: res }
    }

    fn degree(&self) -> u8 {
        self.data.keys().map(|monomial| monomial.len()).max().unwrap_or_default()
    }

    fn is_real(&self) -> bool {
        false
    }
}

impl AdjointTrait for RustComplexCoefficientsCommutativePolynomial {
    fn adjoint(&self) -> Self {
        Self { data: self.data.iter().map(|(monomial, coeff)| (monomial.adjoint(), coeff.conj())).collect() }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use num_complex::Complex;
    use num_traits::{Pow, Zero};
    use rstest::{fixture, rstest};

    use super::*;
    use crate::polynomials::commutative_polynomials::operators::commutative_operator::RustCommutativeOperator;

    #[fixture]
    fn poly() -> RustComplexCoefficientsCommutativePolynomial {
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]),
                        0,
                    ),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]),
                        0,
                    ),
                    Complex::I,
                ),
            ]),
        }
    }

    #[rstest]
    fn test_adjoint(poly: RustComplexCoefficientsCommutativePolynomial) {
        let adjoint = poly.adjoint();
        assert_eq!(
            adjoint,
            RustComplexCoefficientsCommutativePolynomial {
                data: BTreeMap::from([
                    (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: -2.0 }),
                    (
                        RustCommutativeMonomial::new(
                            BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]),
                            0
                        ),
                        Complex::ONE,
                    ),
                    (
                        RustCommutativeMonomial::new(
                            BTreeMap::from([
                                (RustCommutativeOperator::new('y', 1, true, false, false, 0), 1),
                                (RustCommutativeOperator::new('z', 2, false, false, false, 0), 3),
                            ]),
                            0
                        ),
                        Complex { re: 0.0, im: -1.0 },
                    ),
                ]),
            }
        );
    }

    #[rstest]
    #[case(Complex { re: 0.0, im: 0.0 }, RustComplexCoefficientsCommutativePolynomial::zero())]
    #[case(
        Complex { re: 1.0, im: 2.0 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: -3.0, im: 4.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex { re: -2.0, im: 1.0 },
                ),
            ]),
        }
    )]
    fn test_mul_complex(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly * rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('y', 1, false, false, false, 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('y', 1, false, false, false, 0), 1)]), 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                        ]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 2),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_mul_complex_operator(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&poly * &rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
                (RustCommutativeOperator::new('a', 1, false, false, false, 0), 1),
                (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
            ]), 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('a', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex { re: 1.0, im: 2.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('a', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('a', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 6),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_mul_complex_monomial(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&poly * &rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(RustComplexCoefficientsCommutativePolynomial::zero(), RustComplexCoefficientsCommutativePolynomial::zero())]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, false, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: -3.0, im: 4.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: 2.0, im: 4.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex { re: -2.0, im: 1.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 2)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, false, false, false, 0), 3),
                        ]), 0),
                    Complex { re: -2.0, im: 1.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('x', 2, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, false, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 2),
                            (RustCommutativeOperator::new('z', 2, false, false, false, 0), 3),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
            ]),
        }
    )]
    fn test_mul_complex_polynomial(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&poly * &rhs).unwrap(), expected);
        assert_eq!((&poly * rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex { re: 0.0, im: 0.0 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: 1.5, im: 3.5 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 2.5, im: 5.5 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: -1.0, im: -2.0 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_complex(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&poly + rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        Complex { re: 0.0, im: 0.0 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: 1.5, im: 3.5 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: -0.5, im: -1.5 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        Complex { re: 1.0, im: 2.0 },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_sub_complex(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: Complex<f64>,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!((&poly - rhs).unwrap(), expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 2, false, false, false, 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeOperator::new('x', 2, true, false, false, 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_complex_operator(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly + &rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeOperator::new('x', 2, false, false, false, 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeOperator::new('x', 2, true, false, false, 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]), 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_sub_complex_operator(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustCommutativeOperator,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly - &rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 2.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]), 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_complex_monomial(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly + &rhs, expected);
    }

    #[rstest]
    #[case(
        RustCommutativeMonomial::one(0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 0.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]), 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]), 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustCommutativeMonomial::new(BTreeMap::from([
                (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
            ]), 0),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex { re: -1.0, im: 1.0 },
                ),
            ]),
        }
    )]
    fn test_sub_complex_monomial(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustCommutativeMonomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly - &rhs, expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsCommutativePolynomial::zero(),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 2),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 2.0, im: 4.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 2),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    fn test_add_complex_polynomial(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly + &rhs, expected);
        assert_eq!(&poly + rhs, expected);
    }

    #[rstest]
    #[case(
        RustComplexCoefficientsCommutativePolynomial::zero(),
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex::ONE,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        }
    )]
    #[case(
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: -1.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 2),
                        ]), 0),
                    Complex::I,
                ),
            ]),
        },
        RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (
                    RustCommutativeMonomial::new(BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]), 0),
                    Complex { re: 2.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]), 0),
                    Complex::I,
                ),
                (
                    RustCommutativeMonomial::new(BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 2),
                        ]), 0),
                    Complex { re: 0.0, im: -1.0 },
                ),
            ]),
        }
    )]
    fn test_sub_complex_polynomial(
        poly: RustComplexCoefficientsCommutativePolynomial,
        #[case] rhs: RustComplexCoefficientsCommutativePolynomial,
        #[case] expected: RustComplexCoefficientsCommutativePolynomial,
    ) {
        assert_eq!(&poly - &rhs, expected);
        assert_eq!(&poly - rhs, expected);
    }

    #[rstest]
    fn test_neg(poly: RustComplexCoefficientsCommutativePolynomial) {
        let expected = RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: -1.0, im: -2.0 }),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, false, false, false, 0), 1)]),
                        0,
                    ),
                    Complex { re: -1.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([
                            (RustCommutativeOperator::new('y', 1, false, false, false, 0), 1),
                            (RustCommutativeOperator::new('z', 2, true, false, false, 0), 3),
                        ]),
                        0,
                    ),
                    Complex { re: 0.0, im: -1.0 },
                ),
            ]),
        };
        assert_eq!(-&poly, expected);
        assert_eq!(-poly, expected);
    }

    #[test]
    fn test_pow() {
        let poly = RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex { re: 1.0, im: 2.0 }),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]),
                        0,
                    ),
                    Complex::ONE,
                ),
            ]),
        };
        let expected = RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([(RustCommutativeMonomial::one(0), Complex::ONE)]),
        };
        assert_eq!((&poly).pow(0u8).unwrap(), expected);
        assert_eq!((&poly).pow(1u8).unwrap(), poly.clone());

        let poly = RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex::ONE),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]),
                        0,
                    ),
                    Complex::I,
                ),
            ]),
        };
        let expected = RustComplexCoefficientsCommutativePolynomial {
            data: BTreeMap::from([
                (RustCommutativeMonomial::one(0), Complex::ONE),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 1)]),
                        0,
                    ),
                    Complex { re: 0.0, im: 7.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 2)]),
                        0,
                    ),
                    Complex { re: -21.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 3)]),
                        0,
                    ),
                    Complex { re: 0.0, im: -35.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 4)]),
                        0,
                    ),
                    Complex { re: 35.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 5)]),
                        0,
                    ),
                    Complex { re: 0.0, im: 21.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 6)]),
                        0,
                    ),
                    Complex { re: -7.0, im: 0.0 },
                ),
                (
                    RustCommutativeMonomial::new(
                        BTreeMap::from([(RustCommutativeOperator::new('x', 2, true, false, false, 0), 7)]),
                        0,
                    ),
                    Complex { re: 0.0, im: -1.0 },
                ),
            ]),
        };
        assert_eq!((&poly).pow(7u8).unwrap(), expected);
    }
}
