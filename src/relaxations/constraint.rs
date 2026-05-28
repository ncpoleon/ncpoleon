use num_complex::Complex;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::RustCommutativeMonomial;
use crate::polynomials::commutative_polynomials::polynomials::commutative_polynomial::{
    PythonComplexCoefficientsCommutativePolynomial, PythonRealCoefficientsCommutativePolynomial,
};
use crate::polynomials::monomial::{Monomial, OneWithMomentMatrixId};
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::RustNonCommutativeMonomial;
use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::{
    PythonComplexCoefficientsNonCommutativePolynomial, PythonRealCoefficientsNonCommutativePolynomial,
};
use crate::polynomials::polynomial::{Polynomial, PolynomialDtype};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConstraintKind {
    Equality,
    Inequality,
}

#[derive(Clone)]
pub(crate) enum ConstraintSide<MonomialType: Ord, Scalar: PolynomialDtype> {
    Polynomial(Polynomial<MonomialType, Scalar>),
    Scalar(Scalar),
}

#[derive(Clone)]
pub(crate) struct Constraint<MonomialType: Ord, Scalar: PolynomialDtype> {
    pub(crate) kind: ConstraintKind,
    pub(crate) lhs: ConstraintSide<MonomialType, Scalar>,
    pub(crate) rhs: ConstraintSide<MonomialType, Scalar>,
}

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "RealCoefficientsCommutativeConstraint")]
#[derive(Clone)]
pub(crate) struct PythonRealCoefficientsCommutativeConstraint(pub(crate) Constraint<RustCommutativeMonomial, f64>);

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "ComplexCoefficientsCommutativeConstraint")]
#[derive(Clone)]
pub(crate) struct PythonComplexCoefficientsCommutativeConstraint(
    pub(crate) Constraint<RustCommutativeMonomial, Complex<f64>>,
);

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "RealCoefficientsNonCommutativeConstraint")]
#[derive(Clone)]
pub(crate) struct PythonRealCoefficientsNonCommutativeConstraint(
    pub(crate) Constraint<RustNonCommutativeMonomial, f64>,
);

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "ComplexCoefficientsNonCommutativeConstraint")]
#[derive(Clone)]
pub(crate) struct PythonComplexCoefficientsNonCommutativeConstraint(
    pub(crate) Constraint<RustNonCommutativeMonomial, Complex<f64>>,
);

impl<MonomialType: Ord + Clone> From<ConstraintSide<MonomialType, f64>> for ConstraintSide<MonomialType, Complex<f64>>
where
    Polynomial<MonomialType, Complex<f64>>: From<Polynomial<MonomialType, f64>>,
{
    fn from(value: ConstraintSide<MonomialType, f64>) -> Self {
        match value {
            ConstraintSide::Polynomial(poly) => ConstraintSide::Polynomial(poly.into()),
            ConstraintSide::Scalar(s) => ConstraintSide::Scalar(Complex::from(s)),
        }
    }
}

impl<MonomialType: Ord + Clone> From<Constraint<MonomialType, f64>> for Constraint<MonomialType, Complex<f64>>
where
    Polynomial<MonomialType, Complex<f64>>: From<Polynomial<MonomialType, f64>>,
{
    fn from(value: Constraint<MonomialType, f64>) -> Self {
        Constraint { kind: value.kind, lhs: value.lhs.into(), rhs: value.rhs.into() }
    }
}

impl<Data: Clone + Ord, Scalar: PolynomialDtype> Constraint<Monomial<Data>, Scalar> {
    pub(super) fn into_polynomial_diff(self) -> Result<Polynomial<Monomial<Data>, Scalar>, String>
    where
        Monomial<Data>: OneWithMomentMatrixId,
    {
        match (self.lhs, self.rhs) {
            (ConstraintSide::Polynomial(poly_left), ConstraintSide::Polynomial(poly_right)) => {
                Ok(poly_left - poly_right)
            }
            (ConstraintSide::Polynomial(poly_left), ConstraintSide::Scalar(scalar_right)) => poly_left - scalar_right,
            (ConstraintSide::Scalar(scalar_left), ConstraintSide::Polynomial(poly_right)) => -poly_right + scalar_left,
            (ConstraintSide::Scalar(_), ConstraintSide::Scalar(_)) => {
                Err("Both sides of an operator constraint cannot be scalars.".to_string())
            }
        }
    }

    pub(super) fn into_poly_scalar_tuple(self) -> Result<(Polynomial<Monomial<Data>, Scalar>, Scalar), String>
    where
        Monomial<Data>: OneWithMomentMatrixId,
    {
        match (self.lhs, self.rhs) {
            (ConstraintSide::Polynomial(_), ConstraintSide::Polynomial(_)) => {
                Err("Cannot convert this constraint to a (polynomial, scalar) form as both sides are polynomials."
                    .to_string())
            }
            (ConstraintSide::Polynomial(poly_left), ConstraintSide::Scalar(scalar_right)) => {
                Ok((poly_left, scalar_right))
            }
            (ConstraintSide::Scalar(scalar_left), ConstraintSide::Polynomial(poly_right)) => match self.kind {
                ConstraintKind::Equality => Ok((poly_right, scalar_left)),
                ConstraintKind::Inequality => Ok((-poly_right, -scalar_left)),
            },
            (ConstraintSide::Scalar(_), ConstraintSide::Scalar(_)) => {
                Err("Cannot convert this constraint to a (polynomial, scalar) form as both sides are scalars."
                    .to_string())
            }
        }
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonRealCoefficientsCommutativeConstraint {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(c) = value.cast::<PythonRealCoefficientsCommutativeConstraint>() {
            Ok(c.get().clone())
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonRealCoefficientsCommutativeConstraint"))
        }
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonComplexCoefficientsCommutativeConstraint {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(c) = value.cast::<PythonComplexCoefficientsCommutativeConstraint>() {
            Ok(c.get().clone())
        } else if let Ok(c) = value.cast::<PythonRealCoefficientsCommutativeConstraint>() {
            Ok(PythonComplexCoefficientsCommutativeConstraint(c.get().0.clone().into()))
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonComplexCoefficientsCommutativeConstraint"))
        }
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonRealCoefficientsNonCommutativeConstraint {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(c) = value.cast::<PythonRealCoefficientsNonCommutativeConstraint>() {
            Ok(c.get().clone())
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonRealCoefficientsNonCommutativeConstraint"))
        }
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PythonComplexCoefficientsNonCommutativeConstraint {
    type Error = PyErr;
    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(c) = value.cast::<PythonComplexCoefficientsNonCommutativeConstraint>() {
            Ok(c.get().clone())
        } else if let Ok(c) = value.cast::<PythonRealCoefficientsNonCommutativeConstraint>() {
            Ok(PythonComplexCoefficientsNonCommutativeConstraint(c.get().0.clone().into()))
        } else {
            Err(PyTypeError::new_err("Couldn't convert to PythonComplexCoefficientsNonCommutativeConstraint"))
        }
    }
}

macro_rules! impl_constraint_pymethods {
    ($py_constraint:ident, $py_poly:ident) => {
        #[pymethods]
        impl $py_constraint {
            #[getter]
            fn is_equality(&self) -> bool {
                matches!(self.0.kind, ConstraintKind::Equality)
            }

            #[getter]
            fn is_inequality(&self) -> bool {
                matches!(self.0.kind, ConstraintKind::Inequality)
            }

            #[getter]
            fn lhs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                match &self.0.lhs {
                    ConstraintSide::Polynomial(p) => $py_poly(p.clone()).into_py_any(py),
                    ConstraintSide::Scalar(s) => s.into_py_any(py),
                }
            }

            #[getter]
            fn rhs<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                match &self.0.rhs {
                    ConstraintSide::Polynomial(p) => $py_poly(p.clone()).into_py_any(py),
                    ConstraintSide::Scalar(s) => s.into_py_any(py),
                }
            }

            fn __str__(&self) -> String {
                let op = match self.0.kind {
                    ConstraintKind::Equality => "==",
                    ConstraintKind::Inequality => ">=",
                };
                let format_side = |side: &ConstraintSide<_, _>| match side {
                    ConstraintSide::Polynomial(p) => format!("{}", p),
                    ConstraintSide::Scalar(s) => format!("{}", s),
                };
                format!("Constraint({} {} {})", format_side(&self.0.lhs), op, format_side(&self.0.rhs))
            }

            fn __repr__(&self) -> String {
                self.__str__()
            }
        }
    };
}

impl_constraint_pymethods!(PythonRealCoefficientsCommutativeConstraint, PythonRealCoefficientsCommutativePolynomial);
impl_constraint_pymethods!(
    PythonComplexCoefficientsCommutativeConstraint,
    PythonComplexCoefficientsCommutativePolynomial
);
impl_constraint_pymethods!(
    PythonRealCoefficientsNonCommutativeConstraint,
    PythonRealCoefficientsNonCommutativePolynomial
);
impl_constraint_pymethods!(
    PythonComplexCoefficientsNonCommutativeConstraint,
    PythonComplexCoefficientsNonCommutativePolynomial
);

// Side-extraction helpers. Each tries scalar first (so e.g. `poly == 1.0` keeps the
// 1.0 as a literal scalar instead of lifting it to `1.0 * Identity`), then falls back
// to converting `value` into the corresponding polynomial pyclass.

fn try_into_real_commutative_side<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<ConstraintSide<RustCommutativeMonomial, f64>> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(ConstraintSide::Scalar(scalar));
    }
    let poly = PythonRealCoefficientsCommutativePolynomial::try_from(value)?;
    Ok(ConstraintSide::Polynomial(poly.0))
}

fn try_into_complex_commutative_side<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<ConstraintSide<RustCommutativeMonomial, Complex<f64>>> {
    if let Ok(scalar) = value.extract::<Complex<f64>>() {
        return Ok(ConstraintSide::Scalar(scalar));
    }
    let poly = PythonComplexCoefficientsCommutativePolynomial::try_from(value)?;
    Ok(ConstraintSide::Polynomial(poly.0))
}

fn try_into_real_noncommutative_side<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<ConstraintSide<RustNonCommutativeMonomial, f64>> {
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(ConstraintSide::Scalar(scalar));
    }
    let poly = PythonRealCoefficientsNonCommutativePolynomial::try_from(value)?;
    Ok(ConstraintSide::Polynomial(poly.0))
}

fn try_into_complex_noncommutative_side<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<ConstraintSide<RustNonCommutativeMonomial, Complex<f64>>> {
    if let Ok(scalar) = value.extract::<Complex<f64>>() {
        return Ok(ConstraintSide::Scalar(scalar));
    }
    let poly = PythonComplexCoefficientsNonCommutativePolynomial::try_from(value)?;
    Ok(ConstraintSide::Polynomial(poly.0))
}

/// Build a commutative `Constraint` pyclass for the given `(lhs, rhs)` pair, picking the
/// narrowest scalar type (real ⊂ complex). Callers must already know both sides belong to the
/// commutative family — a non-commutative operand will produce a `PyTypeError`.
pub(crate) fn make_commutative_constraint<'py>(
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    kind: ConstraintKind,
) -> PyResult<Py<PyAny>> {
    let py = lhs.py();

    if let (Ok(lhs_side), Ok(rhs_side)) = (try_into_real_commutative_side(lhs), try_into_real_commutative_side(rhs)) {
        return PythonRealCoefficientsCommutativeConstraint(Constraint { kind, lhs: lhs_side, rhs: rhs_side })
            .into_py_any(py);
    }
    if let (Ok(lhs_side), Ok(rhs_side)) =
        (try_into_complex_commutative_side(lhs), try_into_complex_commutative_side(rhs))
    {
        return PythonComplexCoefficientsCommutativeConstraint(Constraint { kind, lhs: lhs_side, rhs: rhs_side })
            .into_py_any(py);
    }

    Err(PyTypeError::new_err(
        "Couldn't build a commutative Constraint from the provided operands. \
         Both sides must be coercible to a commutative polynomial or scalar.",
    ))
}

/// Build a non-commutative `Constraint` pyclass for the given `(lhs, rhs)` pair, picking the
/// narrowest scalar type (real ⊂ complex). Callers must already know both sides belong to the
/// non-commutative family — a commutative operand will produce a `PyTypeError`.
pub(crate) fn make_noncommutative_constraint<'py>(
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    kind: ConstraintKind,
) -> PyResult<Py<PyAny>> {
    let py = lhs.py();

    if let (Ok(lhs_side), Ok(rhs_side)) =
        (try_into_real_noncommutative_side(lhs), try_into_real_noncommutative_side(rhs))
    {
        return PythonRealCoefficientsNonCommutativeConstraint(Constraint { kind, lhs: lhs_side, rhs: rhs_side })
            .into_py_any(py);
    }
    if let (Ok(lhs_side), Ok(rhs_side)) =
        (try_into_complex_noncommutative_side(lhs), try_into_complex_noncommutative_side(rhs))
    {
        return PythonComplexCoefficientsNonCommutativeConstraint(Constraint { kind, lhs: lhs_side, rhs: rhs_side })
            .into_py_any(py);
    }

    Err(PyTypeError::new_err(
        "Couldn't build a non-commutative Constraint from the provided operands. \
         Both sides must be coercible to a non-commutative polynomial or scalar.",
    ))
}
