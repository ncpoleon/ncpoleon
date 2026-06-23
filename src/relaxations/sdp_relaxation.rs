use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::iter::repeat_n;
use std::ops::Mul;

use itertools::Itertools;
use log::{debug, info, trace};
use num_complex::Complex;
use num_traits::Zero;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::{
    PythonCommutativeMonomial, RustCommutativeMonomial,
};
use crate::polynomials::commutative_polynomials::operators::commutative_operator::PythonCommutativeOperator;
use crate::polynomials::commutative_polynomials::polynomials::commutative_polynomial::{
    PythonComplexCoefficientsCommutativePolynomial, PythonRealCoefficientsCommutativePolynomial,
};
use crate::polynomials::monomial::{
    AdjointTrait, HasAMomentMatrixId, Monomial, OneWithMomentMatrixId, RewritingStrategy, RewritingTrait,
};
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::{
    PythonNonCommutativeMonomial, RustNonCommutativeMonomial,
};
use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::PythonNonCommutativeOperator;
use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::{
    PythonComplexCoefficientsNonCommutativePolynomial, PythonRealCoefficientsNonCommutativePolynomial,
};
use crate::polynomials::polynomial::{Polynomial, PolynomialDtype, PolynomialTrait, TryIntoReal};
use crate::relaxations::constraint::{
    ConstraintKind, PythonComplexCoefficientsCommutativeConstraint, PythonComplexCoefficientsNonCommutativeConstraint,
    PythonRealCoefficientsCommutativeConstraint, PythonRealCoefficientsNonCommutativeConstraint,
};
use crate::relaxations::moment_matrix::{
    PythonComplexValuedCommutativeMomentMatrix, PythonComplexValuedNonCommutativeMomentMatrix,
    PythonRealValuedCommutativeMomentMatrix, PythonRealValuedNonCommutativeMomentMatrix, RustMomentMatrix,
};

/// Macro to convert Python polynomial objects into Rust types, create an SDP relaxation, and wrap
/// it in the appropriate Python class. Parameterized by the Python polynomial type and relaxation
/// wrapper type.
macro_rules! build_relaxation_inner {
    (
        $py:expr, $level:expr, $objective:expr,
        $operator_constraints_some:expr, $moment_constraints_some:expr, $normalization_constraints_some:expr,
        $variables:expr, $substitutions:expr, $strategy:expr,
        $py_poly:ident, $py_relaxation:ident, $py_constraint:ident $(,)?
    ) => {{
        let rust_objective = match $py_poly::try_from($objective) {
            Ok(polynomial) => polynomial.0,
            Err(_) => {
                return Err(PyValueError::new_err(concat!(
                    "Couldn't convert objective polynomial to ",
                    stringify!($py_poly)
                )));
            }
        };

        let mut rust_equalities = Vec::with_capacity($operator_constraints_some.len());
        let mut rust_inequalities = Vec::with_capacity($operator_constraints_some.len());
        debug!("Converting operator constraints.");
        for (index, value) in $operator_constraints_some.iter().enumerate() {
            let constraint = $py_constraint::try_from(&value)
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        concat!(
                            "Couldn't convert operator constraint at index {} into a ",
                            stringify!($py_constraint),
                            ". Operator constraints must be ",
                            stringify!($py_constraint),
                            " instances built from `==`/`<=`/`>=` between operators/monomials/polynomials."
                        ),
                        index
                    ))
                })?
                .0;
            let kind = constraint.kind;
            let diff = constraint.into_polynomial_diff().map_err(PyValueError::new_err)?;
            match kind {
                ConstraintKind::Equality => {
                    debug!("Adding polynomial at index {} to the equalities. ({})", index, diff);
                    rust_equalities.push(diff);
                }
                ConstraintKind::Inequality => {
                    debug!("Adding polynomial at index {} to the inequalities. ({})", index, diff);
                    rust_inequalities.push(diff);
                }
            }
        }

        let mut rust_moment_equalities = Vec::new();
        let mut rust_moment_inequalities = Vec::new();
        debug!("Converting moment constraints.");
        for (index, value) in $moment_constraints_some.iter().enumerate() {
            let constraint = $py_constraint::try_from(&value)
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        concat!(
                            "Couldn't convert moment constraint at index {} into a ",
                            stringify!($py_constraint),
                            "."
                        ),
                        index
                    ))
                })?
                .0;
            let kind = constraint.kind;
            let (poly, scalar) = constraint.into_poly_scalar_tuple().map_err(PyValueError::new_err)?;
            match kind {
                ConstraintKind::Equality => {
                    debug!("Adding moment constraints at index {} to the moment constraints equalities. ({} == {})", index, poly, scalar);
                    rust_moment_equalities.push((poly, scalar));
                }
                ConstraintKind::Inequality => {
                    debug!("Adding moment constraints at index {} to the moment constraints inequalities. ({} >= {})", index, poly, scalar);
                    rust_moment_inequalities.push((poly, scalar.try_into_real().map_err(PyValueError::new_err)?));
                }
            }
        }

        let mut rust_normalization_equalities = Vec::new();
        let mut rust_normalization_inequalities = Vec::new();
        debug!("Converting normalization constraints.");
        for (index, value) in $normalization_constraints_some.iter().enumerate() {
            let constraint = $py_constraint::try_from(&value)
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        concat!(
                            "Couldn't convert normalization constraint at index {} into a ",
                            stringify!($py_constraint),
                            "."
                        ),
                        index
                    ))
                })?
                .0;
            let kind = constraint.kind;
            let (poly, scalar) = constraint.into_poly_scalar_tuple().map_err(PyValueError::new_err)?;
            match kind {
                ConstraintKind::Equality => {
                    debug!("Adding normalization constraints at index {} to the normalization constraints equalities. ({} == {})", index, poly, scalar);
                    rust_normalization_equalities.push((poly, scalar));
                }
                ConstraintKind::Inequality => {
                    debug!("Adding normalization constraints at index {} to the normalization constraints inequalities. ({} >= {})", index, poly, scalar);
                    rust_normalization_inequalities.push((poly, scalar.try_into_real().map_err(PyValueError::new_err)?));
                }
            }
        }

        let mut relaxation = SdpRelaxation::new($strategy);
        info!("Setting relaxation.");
        relaxation.set_relaxation(
            $level,
            $variables,
            rust_objective,
            $substitutions,
            rust_equalities,
            rust_inequalities,
            rust_moment_equalities,
            rust_moment_inequalities,
            rust_normalization_equalities,
            rust_normalization_inequalities,
        )?;
        $py_relaxation(relaxation).into_py_any($py)
    }};
}

/// Macro to handle the full conversion pipeline for a given monomial type: parse substitutions,
/// branch on real vs complex, and call `build_relaxation_inner!`.
macro_rules! build_relaxation_arm {
    (
        $py:expr, $level:expr, $objective:expr,
        $operator_constraints_some:expr, $moment_constraints_some:expr, $normalization_constraints_some:expr,
        $substitutions_some:expr, $substitution_strategy:expr,
        monomials: $py_monomial:ident & $rust_monomial:ty,
        variables: $variables:expr,
        real_poly_and_relaxation: $real_py_poly:ident & $real_py_relaxation:ident & $real_py_constraint:ident,
        complex_poly_and_relaxation: $complex_py_poly:ident & $complex_py_relaxation:ident & $complex_py_constraint:ident,
        is_real: $is_real:expr $(,)?
    ) => {{
        let mut rust_substitutions: BTreeMap<$rust_monomial, $rust_monomial> = BTreeMap::new();

        debug!("Converting substitutions.");
        for (index, (monom_key, monom_value)) in $substitutions_some.iter().enumerate() {
            let try_rust_monom_key = $py_monomial::try_from(monom_key);
            let try_rust_monom_value = $py_monomial::try_from(monom_value);

            match (try_rust_monom_key, try_rust_monom_value) {
                (Ok(key), Ok(mut value)) => {
                    // If the RHS term is the identity, it may have been converted from 1, in which
                    // case the conversion couldn't know the moment_matrix index. We set it to the
                    // same one as the monomial to replace
                    if value.0.is_one() {
                        debug!("Set the moment matrix index of the identity operator to the same one as {} in a substitution constraint.", key.0);
                        value.0.data.moment_matrix_id = key.0.data.moment_matrix_id;
                    }
                    trace!("Adding substitution at index {} to the substitutions ({} -> {}).", index, key.0, value.0);
                    rust_substitutions.insert(key.0, value.0);
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Couldn't convert substitution at index {} to a monomial.",
                        index
                    )));
                }
            }
        }

        if $is_real {
            debug!("Setting real-valued relaxation.");
            build_relaxation_inner!(
                $py,
                $level,
                $objective,
                $operator_constraints_some,
                $moment_constraints_some,
                $normalization_constraints_some,
                $variables,
                rust_substitutions,
                $substitution_strategy,
                $real_py_poly,
                $real_py_relaxation,
                $real_py_constraint
            )
        } else {
            debug!("Setting complex-valued relaxation.");
            build_relaxation_inner!(
                $py,
                $level,
                $objective,
                $operator_constraints_some,
                $moment_constraints_some,
                $normalization_constraints_some,
                $variables,
                rust_substitutions,
                $substitution_strategy,
                $complex_py_poly,
                $complex_py_relaxation,
                $complex_py_constraint
            )
        }
    }};
}

macro_rules! impl_sdp_relaxation_pymethods {
    ($py_relaxation:ident, $py_poly:ident, $py_monomial:ident, $py_moment_matrix:ident, $scalar:ty) => {
        #[pymethods]
        impl $py_relaxation {
            fn change_variables<'py>(
                &self,
                // FIXME: should probaby use a reference here, otherwise the polynomial is cloned
                polynomial: $py_poly,
                mapping: &Bound<'py, PyDict>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let res = polynomial
                    .0
                    .data
                    .iter()
                    .map(|(mon, &coeff)| {
                        let moment_matrix = self.0.moment_matrices.get(
                            &mon.moment_matrix_id()).ok_or(PyValueError::new_err(format!(
                                "Couldn't find the moment matrix identifier {} associated to the monomial {} in the moment matrices.",
                                mon.moment_matrix_id(),
                                mon
                            ))
                        )?;
                        let (canonical, is_adjoint, is_real) = moment_matrix
                            .get_canonical(mon, self.0.substitution_strategy, &self.0.substitutions)
                            .map_err(PyValueError::new_err)?;
                        let mapped = mapping.get_item($py_monomial(canonical));

                        if let Ok(Some(mapped)) = mapped {
                            if !is_adjoint || is_real {
                                mapped.mul(coeff)
                            } else {
                                mapped.call_method0("conj")?.mul(coeff)
                            }
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

            // FIXME: This docstring is unclear and not helpful
            /// Splits a polynomial of moments into its real and imaginary parts.
            ///
            /// Given `P = Σ_m c_m [m]` where each `[m]` is a (possibly complex) moment, this
            /// groups contributions by canonical monomial and returns two maps:
            ///
            /// - `real_part`: for each canonical monomial `μ` with value `p + qi`, gives `(a, b)`
            ///   such that `Σ_μ (a·p + b·q) = Re(P)`.
            /// - `imag_part`: same structure, giving `(c, d)` such that `Σ_μ (c·p + d·q) = Im(P)`.
            ///
            /// For Hermitian monomials (real-valued moments), the `Option<f64>` is `None` and no
            /// entry appears in `imag_part`. Returns `(real_part, None)` when `imag_part` is empty,
            /// i.e. when the polynomial evaluates to a real number for all moment values.
            ///
            /// Used to extract dual SDP coefficients from moment (in)equality constraints.
            // FIXME: Always returning a complex coefficient, with a value of 0 instead of an Option that is None,
            //  would simplify the code for exporting using the dual problem
            fn split_into_real_and_imaginary_parts(
                &self,
                polynomial: &$py_poly,
            ) -> PyResult<(BTreeMap<$py_monomial, (f64, Option<f64>)>, Option<BTreeMap<$py_monomial, (f64, Option<f64>)>>)> {
                let mut real_part: BTreeMap<_, (f64, Option<f64>)> = BTreeMap::new();
                let mut imag_part: BTreeMap<_, (f64, Option<f64>)> = BTreeMap::new();
                const REALNESS_INCONSISTENCY_ERR: &str =
                    "Canonical monomial inconsistently marked as real and non-real. This is likely an error on our \
                    part, so feel free to open an issue!";

                for (mon, &coeff) in polynomial.0.data.iter() {
                    let moment_matrix = self.0.moment_matrices.get(
                        &mon.moment_matrix_id()).ok_or(PyValueError::new_err(format!(
                            "Couldn't find the moment matrix identifier {} associated to the monomial {} in the moment matrices.",
                            mon.moment_matrix_id(),
                            mon
                        ))
                    )?;
                    // Unify f64/Complex<f64>, so that the macro knows that we deal with complex numbers
                    let as_complex = Complex::from(coeff);
                    let (canonical, is_adjoint, is_real) = moment_matrix
                        .get_canonical(mon, self.0.substitution_strategy, &self.0.substitutions)
                        .map_err(PyValueError::new_err)?;

                    if is_real {
                        real_part.entry(canonical).and_modify(|e| (*e).0 += as_complex.re).or_insert((as_complex.re, None));
                    } else {
                        real_part
                            .entry(canonical.clone())
                            .and_modify(|e| {
                                (*e).0 += as_complex.re;
                                let imag = e.1.as_mut().expect(REALNESS_INCONSISTENCY_ERR);
                                *imag += if is_adjoint { as_complex.im } else { -as_complex.im };
                            })
                            .or_insert((as_complex.re, Some(if is_adjoint {as_complex.im} else {-as_complex.im})));
                        imag_part
                            .entry(canonical)
                            .and_modify(|e| {
                                (*e).0 += as_complex.im;
                                let imag = e.1.as_mut().expect(REALNESS_INCONSISTENCY_ERR);
                                *imag += if is_adjoint { -as_complex.re } else { as_complex.re };
                            })
                            .or_insert((as_complex.im, Some(if is_adjoint {-as_complex.re} else {as_complex.re})));
                    }
                }

                // FiXME: slightly slower to do it like this, but much cleaner to code. Benchmark whether
                //  adding and then filtering is much slower than not adding/removing if coeff is nil
                let python_real_part = real_part
                    .into_iter()
                    .filter(|(_mon, (coeff_re, coeff_im))| match coeff_im {
                        None => *coeff_re != 0.0,
                        Some(coeff_im) => (*coeff_re != 0.0) || (*coeff_im != 0.0)
                    })
                    .map(|(rust_monomial, coeff)| ($py_monomial(rust_monomial), coeff))
                    .collect();
                let python_imag_part: BTreeMap<_, _> = imag_part
                    .into_iter()
                    // We use unwrap here since we always insert the imaginarity part with Some, no None is unreachable
                    .filter(|(_mon, (coeff_re, coeff_im))| *coeff_re != 0.0 || coeff_im.unwrap() != 0.0)
                    .map(|(rust_monomial, coeff)| ($py_monomial(rust_monomial), coeff))
                    .collect();

                if python_imag_part.is_empty() {
                    Ok((python_real_part, None))
                } else {
                    Ok((python_real_part, Some(python_imag_part)))
                }
            }

            #[getter]
            fn is_real(&self) -> bool {
                self.0.objective.is_real()
            }

            #[getter]
            fn objective(&self) -> $py_poly {
                $py_poly(self.0.objective.clone())
            }

            /// Dictionary of all moment matrices.
            ///
            /// Each element corresponds to a unique moment matrix index.
            #[getter]
            fn moment_matrices(&self) -> BTreeMap<u8, $py_moment_matrix> {
                self.0
                    .moment_matrices
                    .iter()
                    .map(|(&index, moment_matrix)| (index, $py_moment_matrix(moment_matrix.clone())))
                    .collect()
            }

            fn rewrite<'py>(&self, mon_or_poly: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
                let py = mon_or_poly.py();
                if let Ok(mon) = TryInto::<$py_monomial>::try_into(mon_or_poly) {
                    $py_monomial(
                        mon.0
                        .rewrite(
                            self.0.substitution_strategy,
                            &self.0.substitutions
                        )
                        .map_err(PyValueError::new_err)?
                    ).into_py_any(py)
                } else {
                    let poly: $py_poly = mon_or_poly.try_into()?;
                    $py_poly(
                        poly.0
                        .rewrite(
                            self.0.substitution_strategy,
                            &self.0.substitutions
                        )
                        .map_err(PyValueError::new_err)?
                    ).into_py_any(py)
                }
            }

            /// Dictionary of all generating sets
            ///
            /// Each element corresponds to a unique moment matrix index.
            #[getter]
            fn generating_sets(&self) -> BTreeMap<u8, Vec<$py_monomial>> {
                self.0
                    .generating_sets
                    .iter()
                    .map(|(&index, generating_set)| (index, generating_set.iter().cloned().map($py_monomial).collect()))
                    .collect()
            }

            /// Localising moment matrices for the inequality constraints.
            ///
            /// These matrices are ordered in a dictionary, with the key being the moment matrix identifier
            /// they are associated with. The values of this dictionary are lists of moment matrices associated
            /// with operator inequalities.
            #[getter]
            fn localising_moment_matrices_inequalities(&self) -> BTreeMap<u8, Vec<$py_moment_matrix>> {
                self.0
                    .localising_moment_matrices_inequalities
                    .iter()
                    .map(|(&index, inequalities)| (index, inequalities.iter().map(|moment_matrix| $py_moment_matrix(moment_matrix.clone())).collect()))
                    .collect()
            }

            /// Localising moment matrices for the equality constraints.
            ///
            /// Same structure as `localising_moment_matrices_inequalities` but
            /// for each equality constraint polynomial.
            #[getter]
            fn localising_moment_matrices_equalities(&self) -> BTreeMap<u8, Vec<$py_moment_matrix>> {
                self.0
                    .localising_moment_matrices_equalities
                    .iter()
                    .map(|(&index, equalities)| (index, equalities.iter().map(|moment_matrix| $py_moment_matrix(moment_matrix.clone())).collect()))
                    .collect()
            }

            /// Moment equality constraints as a list of `(polynomial, value)` pairs, each
            /// encoding `<polynomial> = value`.
            #[getter]
            fn moment_equalities(&self) -> Vec<($py_poly, $scalar)> {
                self.0.moment_equalities.iter().map(|(poly, value)| ($py_poly(poly.clone()), *value)).collect()
            }

            /// Moment inequality constraints as a list of `(polynomial, value)` pairs, each
            /// encoding `<polynomial> >= value`.
            #[getter]
            fn moment_inequalities(&self) -> Vec<($py_poly, f64)> {
                self.0.moment_inequalities.iter().map(|(poly, value)| ($py_poly(poly.clone()), *value)).collect()
            }

            #[getter]
            fn equalities(&self) -> BTreeMap<u8, Vec<$py_poly>> {
                self.0.
                    equalities.
                    iter()
                    .map(|(&mm_id, equalities_id)| (mm_id,equalities_id.iter().map(|poly| $py_poly(poly.clone())).collect()))
                    .collect()
            }

            #[getter]
            fn inequalities(&self) -> BTreeMap<u8, Vec<$py_poly>> {
                self.0
                    .inequalities
                    .iter()
                    .map(|(&mm_id, equalities_id)| (mm_id, equalities_id.iter().map(|poly| $py_poly(poly.clone())).collect()))
                    .collect()
            }
        }
    };
}

// FIXME: Potentially redundant with the TryFrom trait. More generally, we check if it can be cast, and
// then recheck later, this could be optimized
fn is_bound_a_real_valued_polynomial<'py>(bound: &Bound<'py, PyAny>, name: &str) -> PyResult<bool> {
    if bound.extract::<f64>().is_ok()
        || bound.cast::<PythonCommutativeOperator>().is_ok()
        || bound.cast::<PythonNonCommutativeOperator>().is_ok()
        || bound.cast::<PythonCommutativeMonomial>().is_ok()
        || bound.cast::<PythonNonCommutativeMonomial>().is_ok()
        || bound.cast::<PythonRealCoefficientsCommutativePolynomial>().is_ok()
        || bound.cast::<PythonRealCoefficientsNonCommutativePolynomial>().is_ok()
    {
        Ok(true)
    } else if bound.cast::<PythonComplexCoefficientsCommutativePolynomial>().is_ok()
        || bound.cast::<PythonComplexCoefficientsNonCommutativePolynomial>().is_ok()
        || bound.extract::<Complex<f64>>().is_ok()
    {
        Ok(false)
    } else {
        Err(PyValueError::new_err(format!("Couldn't convert {} to a supported Polynomial", name)))
    }
}

pub(crate) fn is_constraint_real_valued<'py>(bound: &Bound<'py, PyAny>, name: &str) -> PyResult<bool> {
    if bound.cast::<PythonRealCoefficientsCommutativeConstraint>().is_ok()
        || bound.cast::<PythonRealCoefficientsNonCommutativeConstraint>().is_ok()
    {
        Ok(true)
    } else if bound.cast::<PythonComplexCoefficientsCommutativeConstraint>().is_ok()
        || bound.cast::<PythonComplexCoefficientsNonCommutativeConstraint>().is_ok()
    {
        Ok(false)
    } else {
        Err(PyValueError::new_err(format!("Couldn't convert {} to a supported constraint", name)))
    }
}

/// Build an SDP relaxation for a (non)commutative polynomial optimisation problem.
///
/// Given a list of operator variables, a relaxation level, and an objective
/// polynomial, this function constructs the moment matrix and localising
/// matrices at the requested level and returns a typed SDP relaxation object.
///
/// # Arguments
/// * `variables` – List of [`CommutativeOperator`] **or** [`NonCommutativeOperator`] instances (mixing the two is not
///   supported yet).
/// * `level` – Level of the relaxation.
/// * `objective` – The polynomial to optimize.
/// * `substitutions` – Optional dictionary mapping monomials to their replacements. For equalities between monomials,
///   `substitutions` should be preferred as it leads to smaller relaxations.
/// * `operator_constraints` – Optional list of `Constraint` objects expressing operator-level equalities and
///   inequalities (e.g. `op == 0`, `op >= 0`).
/// * `moment_constraints` – Optional list of `Constraint` objects expressing moment-level constraints (`<polynomial> ==
///   value` or `<polynomial> >= value`).
/// * `normalization_constraints` – Optional list of `Constraint` objects expressing normalization constraints (e.g.
///   `I_k == 0.5`). For each moment-matrix index `k` not covered by a normalization constraint, the default `<I_k> = 1`
///   is auto-injected.
/// * `substitution_strategy` – How to apply the substitution rules (default: `RewritingStrategy.Greedy`).
/// * `assume_real` – If `True`, the function assumes that the problem is real-valued, instead of trying to infer
///   whether it is the case by trying to covnert every polynomial to a real-valued one. Set this argument to `True` to
///   speed up the initial step of the relaxation if you know that your problem is real-valued.
///
/// # Errors
/// Raises `ValueError` if the variables list is empty, if a variable cannot
/// be identified as commutative or non-commutative, or if any polynomial
/// cannot be converted to the inferred coefficient type.
#[pyfunction]
#[pyo3(
    signature=(
        variables,
        level,
        objective,
        *,
        substitutions=None,
        operator_constraints=None,
        moment_constraints=None,
        normalization_constraints=None,
        substitution_strategy=RewritingStrategy::Greedy,
        assume_real=false,
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_relaxation<'py>(
    variables: &Bound<'py, PyList>,
    level: u8,
    objective: &Bound<'py, PyAny>,
    substitutions: Option<&Bound<'py, PyDict>>,
    operator_constraints: Option<&Bound<'py, PyList>>,
    moment_constraints: Option<&Bound<'py, PyList>>,
    normalization_constraints: Option<&Bound<'py, PyList>>,
    substitution_strategy: RewritingStrategy,
    assume_real: bool,
) -> PyResult<Py<PyAny>> {
    let py = objective.py();
    let default_dict = PyDict::new(py);
    let default_list = PyList::empty(py);
    let substitutions_some = substitutions.unwrap_or(&default_dict);
    let operator_constraints_some = operator_constraints.unwrap_or(&default_list);
    let moment_constraints_some = moment_constraints.unwrap_or(&default_list);
    let normalization_constraints_some = normalization_constraints.unwrap_or(&default_list);

    // We first need to check whether all the constraints are real-valued
    let mut is_problem_real_valued = true;

    if !assume_real {
        info!("Checking whether the problem is real-valued.");
        is_problem_real_valued &= is_bound_a_real_valued_polynomial(objective, "objective")?;

        for (label, constraints_list) in [
            ("operator", operator_constraints_some),
            ("moment", moment_constraints_some),
            ("normalization", normalization_constraints_some),
        ] {
            for (index, value) in constraints_list.iter().enumerate() {
                if !is_problem_real_valued {
                    info!("The problem has been found to be complex-valued.");
                    break;
                }
                is_problem_real_valued &= is_constraint_real_valued(
                    &value,
                    format!("the constraint at index {} of the {} constraints", index, label).as_str(),
                )?;
            }
        }
    }

    // We now want to determine the type of monomials that we'll deal with
    let mut commutative_variables = Vec::with_capacity(0);
    let mut noncommutative_variables = Vec::with_capacity(0);

    for (index_variable, variable) in variables.iter().enumerate() {
        if let Ok(cast_var_to_get) = variable.cast::<PythonCommutativeOperator>() {
            let cast_var: PythonCommutativeOperator = *cast_var_to_get.get();
            commutative_variables.push(cast_var.0);
        } else if let Ok(cast_var_to_get) = variable.cast::<PythonNonCommutativeOperator>() {
            let cast_var: PythonNonCommutativeOperator = *cast_var_to_get.get();
            noncommutative_variables.push(cast_var.0);
        } else {
            return Err(PyValueError::new_err(format!(
                "Couldn't convert variable at index {} into a valid operator.",
                index_variable
            )));
        }
    }

    match (commutative_variables.is_empty(), noncommutative_variables.is_empty()) {
        (true, true) => Err(PyValueError::new_err("Variables must be provided.")),
        // Noncommutative problem
        (true, false) => {
            build_relaxation_arm!(
                py, level, objective,
                operator_constraints_some, moment_constraints_some, normalization_constraints_some,
                substitutions_some, substitution_strategy,
                monomials: PythonNonCommutativeMonomial & RustNonCommutativeMonomial,
                variables: noncommutative_variables,
                real_poly_and_relaxation: PythonRealCoefficientsNonCommutativePolynomial &
                    PythonRealValuedNonCommutativeSdpRelaxation &
                    PythonRealCoefficientsNonCommutativeConstraint,
                complex_poly_and_relaxation: PythonComplexCoefficientsNonCommutativePolynomial &
                    PythonComplexValuedNonCommutativeSdpRelaxation &
                    PythonComplexCoefficientsNonCommutativeConstraint,
                is_real: is_problem_real_valued,
            )
        }
        // Commutative problem
        (false, true) => {
            build_relaxation_arm!(
                py, level, objective,
                operator_constraints_some, moment_constraints_some, normalization_constraints_some,
                substitutions_some, substitution_strategy,
                monomials: PythonCommutativeMonomial & RustCommutativeMonomial,
                variables: commutative_variables,
                real_poly_and_relaxation: PythonRealCoefficientsCommutativePolynomial &
                    PythonRealValuedCommutativeSdpRelaxation &
                    PythonRealCoefficientsCommutativeConstraint,
                complex_poly_and_relaxation: PythonComplexCoefficientsCommutativePolynomial &
                    PythonComplexValuedCommutativeSdpRelaxation &
                    PythonComplexCoefficientsCommutativeConstraint,
                is_real: is_problem_real_valued,
            )
        }
        (false, false) => Err(PyNotImplementedError::new_err(
            "Hybrid polynomials are not handled yet, but both commutative and \
                non-commutative operators have been detected.",
        )),
    }
}

pub(super) struct SdpRelaxation<MonomialType: AdjointTrait + Ord, Scalar: PolynomialDtype> {
    objective: Polynomial<MonomialType, Scalar>,
    substitutions: BTreeMap<MonomialType, MonomialType>,
    substitution_strategy: RewritingStrategy,
    equalities: BTreeMap<u8, Vec<Polynomial<MonomialType, Scalar>>>,
    inequalities: BTreeMap<u8, Vec<Polynomial<MonomialType, Scalar>>>,
    moment_equalities: Vec<(Polynomial<MonomialType, Scalar>, Scalar)>,
    moment_inequalities: Vec<(Polynomial<MonomialType, Scalar>, f64)>,
    moment_matrices: BTreeMap<u8, RustMomentMatrix<Scalar, MonomialType>>,
    generating_sets: BTreeMap<u8, Vec<MonomialType>>,
    localising_moment_matrices_equalities: BTreeMap<u8, Vec<RustMomentMatrix<Scalar, MonomialType>>>,
    localising_moment_matrices_inequalities: BTreeMap<u8, Vec<RustMomentMatrix<Scalar, MonomialType>>>,
}

// Commutative type aliases
pub(super) type CommutativeSdpRelaxation<Scalar> = SdpRelaxation<RustCommutativeMonomial, Scalar>;
pub(super) type RealCommutativeSdpRelaxation = CommutativeSdpRelaxation<f64>;
pub(super) type ComplexCommutativeSdpRelaxation = CommutativeSdpRelaxation<Complex<f64>>;

// Noncommutative type aliases
pub(super) type NonCommutativeSdpRelaxation<Scalar> = SdpRelaxation<RustNonCommutativeMonomial, Scalar>;
pub(super) type RealNonCommutativeSdpRelaxation = NonCommutativeSdpRelaxation<f64>;
pub(super) type ComplexNonCommutativeSdpRelaxation = NonCommutativeSdpRelaxation<Complex<f64>>;

/// SDP relaxation for a commutative polynomial optimisation problem with real coefficients.
///
/// Instances are created by calling [`get_relaxation`] with commutative variables, a
/// real-valued objective and real-valued equalities and inequalities.
///
/// Use the `change_variables` method to substitute the abstract monomials with Python objects.
// Commutative Python wrappers
#[pyclass(frozen, module = "ncpoleon.relaxations", subclass, name = "RealValuedCommutativeSdpRelaxation")]
pub(super) struct PythonRealValuedCommutativeSdpRelaxation(RealCommutativeSdpRelaxation);

/// SDP relaxation for a commutative polynomial optimisation problem with complex coefficients.
///
/// Same as [`RealValuedCommutativeSdpRelaxation`] but the objective and all
/// constraint polynomials may have complex coefficients.
#[pyclass(frozen, module = "ncpoleon.relaxations", subclass, name = "ComplexValuedCommutativeSdpRelaxation")]
pub(super) struct PythonComplexValuedCommutativeSdpRelaxation(ComplexCommutativeSdpRelaxation);

/// SDP relaxation for a non-commutative polynomial optimisation problem with real coefficients.
///
/// Instances are created by calling [`get_relaxation`] with non-commutative
/// variables, a real-valued objective and real-valued equality and inequality constraints.
// Noncommutative Python wrappers
#[pyclass(frozen, module = "ncpoleon.relaxations", subclass, name = "RealValuedNonCommutativeSdpRelaxation")]
pub(super) struct PythonRealValuedNonCommutativeSdpRelaxation(RealNonCommutativeSdpRelaxation);

/// SDP relaxation for a non-commutative polynomial optimisation problem with complex coefficients.
///
/// Same as [`RealValuedNonCommutativeSdpRelaxation`] but the objective and all
/// constraint polynomials may have complex coefficients.
#[pyclass(frozen, module = "ncpoleon.relaxations", subclass, name = "ComplexValuedNonCommutativeSdpRelaxation")]
pub(super) struct PythonComplexValuedNonCommutativeSdpRelaxation(ComplexNonCommutativeSdpRelaxation);

// Generate #[pymethods] for all relaxation wrapper types via macro
impl_sdp_relaxation_pymethods!(
    PythonRealValuedCommutativeSdpRelaxation,
    PythonRealCoefficientsCommutativePolynomial,
    PythonCommutativeMonomial,
    PythonRealValuedCommutativeMomentMatrix,
    f64
);
impl_sdp_relaxation_pymethods!(
    PythonComplexValuedCommutativeSdpRelaxation,
    PythonComplexCoefficientsCommutativePolynomial,
    PythonCommutativeMonomial,
    PythonComplexValuedCommutativeMomentMatrix,
    Complex<f64>
);
impl_sdp_relaxation_pymethods!(
    PythonRealValuedNonCommutativeSdpRelaxation,
    PythonRealCoefficientsNonCommutativePolynomial,
    PythonNonCommutativeMonomial,
    PythonRealValuedNonCommutativeMomentMatrix,
    f64
);
impl_sdp_relaxation_pymethods!(
    PythonComplexValuedNonCommutativeSdpRelaxation,
    PythonComplexCoefficientsNonCommutativePolynomial,
    PythonNonCommutativeMonomial,
    PythonComplexValuedNonCommutativeMomentMatrix,
    Complex<f64>
);

impl<Data: Ord + Clone, Scalar: PolynomialDtype> SdpRelaxation<Monomial<Data>, Scalar>
where
    Polynomial<Monomial<Data>, Scalar>: PolynomialTrait,
    Monomial<Data>: OneWithMomentMatrixId + AdjointTrait,
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    for<'a> Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    for<'a> Monomial<Data>:
        Mul<&'a Polynomial<Monomial<Data>, Scalar>, Output = Result<Polynomial<Monomial<Data>, Scalar>, String>>,
    for<'a> Polynomial<Monomial<Data>, Scalar>:
        Mul<&'a Monomial<Data>, Output = Result<Polynomial<Monomial<Data>, Scalar>, String>>,
{
    pub(super) fn new(substitution_strategy: RewritingStrategy) -> Self {
        Self {
            objective: Polynomial::zero(),
            substitutions: BTreeMap::new(),
            substitution_strategy,
            equalities: BTreeMap::new(),
            inequalities: BTreeMap::new(),
            moment_equalities: Vec::with_capacity(0),
            moment_inequalities: Vec::with_capacity(0),
            moment_matrices: BTreeMap::new(),
            generating_sets: BTreeMap::new(),
            localising_moment_matrices_equalities: BTreeMap::new(),
            localising_moment_matrices_inequalities: BTreeMap::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn set_relaxation<OperatorType: Copy + Ord + AdjointTrait + Display + HasAMomentMatrixId>(
        &mut self,
        level: u8,
        variables: Vec<OperatorType>,
        objective: Polynomial<Monomial<Data>, Scalar>,
        substitutions: BTreeMap<Monomial<Data>, Monomial<Data>>,
        equalities: Vec<Polynomial<Monomial<Data>, Scalar>>,
        inequalities: Vec<Polynomial<Monomial<Data>, Scalar>>,
        moment_equalities: Vec<(Polynomial<Monomial<Data>, Scalar>, Scalar)>,
        moment_inequalities: Vec<(Polynomial<Monomial<Data>, Scalar>, f64)>,
        normalization_equalities: Vec<(Polynomial<Monomial<Data>, Scalar>, Scalar)>,
        normalization_inequalities: Vec<(Polynomial<Monomial<Data>, Scalar>, f64)>,
    ) -> PyResult<()>
    where
        Monomial<Data>: From<OperatorType> + RewritingTrait<Monomial<Data>> + Display,
        Polynomial<Monomial<Data>, Scalar>: RewritingTrait<Monomial<Data>> + Display,
    {
        let mut variables_with_adjoint = BTreeMap::new();

        for variable in variables {
            let adjoint = variable.adjoint();
            match variables_with_adjoint.entry(variable.moment_matrix_id()) {
                Entry::Vacant(empty_entry) => {
                    trace!(
                        "Creating moment matrix with identifier {} and adding {} and its adjoint {} to the variables set.",
                        variable.moment_matrix_id(),
                        variable,
                        adjoint
                    );
                    empty_entry.insert(BTreeSet::from([variable, adjoint]));
                }
                Entry::Occupied(mut occupied_entry) => {
                    let variables_set = occupied_entry.get_mut();
                    trace!("Adding {} to the variables set.", variable);
                    variables_set.insert(variable);
                    trace!("Adding the adjoint {} to the variable set.", adjoint);
                    variables_set.insert(adjoint);
                }
            }
        }

        self.substitutions = substitutions;

        debug!("Partitioning operator equalities constraints.");
        for (index, equality) in equalities.into_iter().enumerate() {
            if let Some(moment_matrix_id) = equality.get_unique_moment_matrix_id() {
                if !variables_with_adjoint.contains_key(&moment_matrix_id) {
                    return Err(PyValueError::new_err(format!(
                        "The polynomial at index {} in the operator equality constraints is defined using the moment matrix identifier {} which isn't associated with a moment matrix.",
                        index, moment_matrix_id
                    )));
                }
                self.equalities.entry(moment_matrix_id).or_default().push(
                    equality.rewrite(self.substitution_strategy, &self.substitutions).map_err(PyValueError::new_err)?,
                );
            } else {
                return Err(PyValueError::new_err(format!(
                    "The polynomial at index {} in the operator equality constraints isn't defined using a unique moment matrix identifier.",
                    index
                )));
            }
        }

        debug!("Partitioning operator inequalities constraints.");
        for (index, inequality) in inequalities.into_iter().enumerate() {
            if let Some(moment_matrix_id) = inequality.get_unique_moment_matrix_id() {
                if !variables_with_adjoint.contains_key(&moment_matrix_id) {
                    return Err(PyValueError::new_err(format!(
                        "The polynomial at index {} in the operator inequality constraints is defined using the moment matrix identifier {} which isn't associated with a moment matrix.",
                        index, moment_matrix_id
                    )));
                }
                self.inequalities.entry(moment_matrix_id).or_default().push(
                    inequality
                        .rewrite(self.substitution_strategy, &self.substitutions)
                        .map_err(PyValueError::new_err)?,
                );
            } else {
                return Err(PyValueError::new_err(format!(
                    "The polynomial at index {} in the operator inequality constraints isn't defined using a unique moment matrix identifier.",
                    index
                )));
            }
        }

        // Auto-inject default normalization `<I_k> = 1` for each moment-matrix index `k` that
        // doesn't already appear in a user-supplied normalization constraint. Only normalization
        // constraints contribute to the "covered" set — generic moment constraints don't, so a user
        // can write `<polynomial> >= c` constraints involving the identity without disabling the
        // default normalization.
        let mut normalization_equalities = normalization_equalities;
        let mut covered_indices = BTreeSet::new();
        for (poly, _) in normalization_equalities.iter() {
            for monomial in poly.data.keys() {
                covered_indices.insert(monomial.moment_matrix_id());
            }
        }
        for (poly, _) in normalization_inequalities.iter() {
            for monomial in poly.data.keys() {
                covered_indices.insert(monomial.moment_matrix_id());
            }
        }
        for &k in variables_with_adjoint.keys() {
            if !covered_indices.contains(&k) {
                debug!("Setting default normalization constraint for the moment matrix at index {}.", k);
                normalization_equalities
                    .push((Polynomial::from(<Monomial<Data> as OneWithMomentMatrixId>::one(k)), Scalar::one()));
            }
        }

        // Merge normalization constraints into the moment lists; from this point on the
        // normalization constraints are indistinguishable from generic moment constraints.
        let mut moment_equalities = moment_equalities;
        moment_equalities.extend(normalization_equalities);
        let mut moment_inequalities = moment_inequalities;
        moment_inequalities.extend(normalization_inequalities);
        self.moment_equalities = moment_equalities
            .into_iter()
            .map(|(poly, scalar)| match poly.rewrite(self.substitution_strategy, &self.substitutions) {
                Ok(rewritten) => Ok((rewritten, scalar)),
                Err(e) => Err(e),
            })
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;
        self.moment_inequalities = moment_inequalities
            .into_iter()
            .map(|(poly, scalar)| match poly.rewrite(self.substitution_strategy, &self.substitutions) {
                Ok(rewritten) => Ok((rewritten, scalar)),
                Err(e) => Err(e),
            })
            .collect::<Result<_, _>>()
            .map_err(PyValueError::new_err)?;

        debug!("Rewriting objective.");
        self.objective =
            objective.rewrite(self.substitution_strategy, &self.substitutions).map_err(PyValueError::new_err)?;

        debug!("Checking if objective is hermitian.");
        // TODO: We could add a chop_delta parameter to chop the resulting polynomial
        let objective_diff = (&self.objective - self.objective.adjoint())
            .rewrite(self.substitution_strategy, &self.substitutions)
            .map_err(PyValueError::new_err)?;
        if objective_diff != Polynomial::zero() {
            return Err(PyValueError::new_err(format!(
                "The objective polynomial must be hermitian but its difference with its adjoint is {}.",
                objective_diff,
            )));
        }

        for (moment_matrix_id, variables_set) in variables_with_adjoint.into_iter() {
            // The i-th element of monomials_sets contains the set of monomials of length i + 1
            // This allows us to access the monomials for lower k_i when dealing with
            // localizing moment matrices
            let mut monomials_sets = Vec::with_capacity(1 + level as usize);
            monomials_sets.push(BTreeSet::from([Monomial::one(moment_matrix_id)]));

            // Generating the monomials set by finding which monomials can be reduced
            // FIXME: if the monomials are commutative, we can instead loop over the possible powers of
            // the operators at a given level, it's way more efficient (comb(d+t-1, t) vs d**t). In
            // order to do so, we could add a is_commutative method to PolynomialTrait, just like we did
            // with is_real. This however wouldn't work to generate Hybrid monomials, we may want to have
            // two different sets of variables, one commutative and one non commutative
            debug!("Generating indexing set for moment matrix identifier {}.", moment_matrix_id);
            for monomial_length in 1..=level {
                debug!("Generating monomials of length {}.", monomial_length);
                let mut level_set = BTreeSet::new();
                repeat_n(variables_set.iter().cloned(), monomial_length as usize)
                    .multi_cartesian_product()
                    .try_for_each(|operators| -> Result<(), String> {
                        let mut iter = operators.into_iter().map(Monomial::from);
                        let first = iter.next().unwrap();
                        let new_monomial = iter.try_fold(first, |acc, m| acc * &m)?;
                        // We remove from the monomials set all monomials that can be reduced via
                        // substitutions
                        trace!("New monomial: {}.", new_monomial);
                        let rewritten = new_monomial.rewrite(self.substitution_strategy, &self.substitutions)?;
                        trace!("Rewritten monomial: {}.", rewritten);
                        // We have to check that a reduced monomial has not been inserted in a previous
                        // level
                        if !level_set.contains(&rewritten)
                            & !monomials_sets.iter().any(|monomial_set| monomial_set.contains(&rewritten))
                        {
                            trace!("Adding the rewritten monomial to the indexing set at level {}.", monomial_length);
                            level_set.insert(rewritten.clone());
                        }
                        Ok(())
                    })
                    .map_err(PyValueError::new_err)?;
                monomials_sets.push(level_set);
            }

            let is_problem_real_valued = self.objective.is_real();
            let mut new_moment_matrix =
                RustMomentMatrix { data: BTreeMap::new(), size: monomials_sets.iter().map(|set| set.len()).sum() };

            // Determine the constraints on the moment matrix. This is where we build the map between
            // reduced monomials and indices within the moment matrix
            for (index_row, monomial_row) in monomials_sets.iter().flatten().enumerate() {
                // FIXME: using skip probably makes it run in n^2 instead of n*(n+1)/2
                for (index_column, monomial_column) in monomials_sets.iter().flatten().enumerate().skip(index_row) {
                    let new_monomial = if index_row == 0 {
                        monomial_column.clone()
                    } else {
                        // FIXME: performance: no need to recompute monomial_row_adjoint within the loop
                        let monomial_row_adjoint = monomial_row.adjoint();
                        (monomial_row_adjoint * monomial_column)
                            .map_err(PyValueError::new_err)?
                            .rewrite(self.substitution_strategy, &self.substitutions)
                            .map_err(PyValueError::new_err)?
                    };

                    if let Some((position_matrix, position_matrix_conj)) = new_moment_matrix
                        .get_mut(&new_monomial, self.substitution_strategy, &self.substitutions)
                        .map_err(PyValueError::new_err)?
                    {
                        position_matrix.insert((index_row, index_column), Scalar::one());
                        if let Some(position_matrix_conj) = position_matrix_conj {
                            position_matrix_conj.insert((index_column, index_row), Scalar::one());
                        } else {
                            position_matrix.insert((index_column, index_row), Scalar::one());
                        }
                        continue;
                    }

                    let use_single_matrix = is_problem_real_valued
                        || (new_monomial
                            == new_monomial
                                .adjoint()
                                .rewrite(self.substitution_strategy, &self.substitutions)
                                .map_err(PyValueError::new_err)?);

                    let new_entry = if use_single_matrix {
                        (
                            BTreeMap::from([
                                ((index_row, index_column), Scalar::one()),
                                // On the diagonal, BTreeMap will remove the extra entry
                                ((index_column, index_row), Scalar::one()),
                            ]),
                            None,
                        )
                    } else {
                        (
                            BTreeMap::from([((index_row, index_column), Scalar::one())]),
                            Some(BTreeMap::from([((index_column, index_row), Scalar::one())])),
                        )
                    };
                    new_moment_matrix.data.insert(new_monomial, new_entry);
                }
            }

            debug!("Computing localizing matrices for equality constraints.");
            // TODO: write a macro/function for equalities and inequalities
            let mut new_localising_moment_matrices_equalities = Vec::with_capacity(self.equalities.len());
            if let Some(equalities) = self.equalities.get(&moment_matrix_id) {
                for equality in equalities.iter() {
                    new_localising_moment_matrices_equalities.push(self.get_localising_moment_matrix(
                        equality,
                        (2 * level - equality.degree()) / 2,
                        &monomials_sets,
                        &new_moment_matrix,
                    )?);
                }
            }
            self.localising_moment_matrices_equalities
                .insert(moment_matrix_id, new_localising_moment_matrices_equalities);

            debug!("Computing localizing matrices for inequality constraints.");
            let mut new_localising_moment_matrices_inequalities = Vec::with_capacity(self.inequalities.len());
            if let Some(inequalities) = self.inequalities.get(&moment_matrix_id) {
                for inequality in inequalities.iter() {
                    new_localising_moment_matrices_inequalities.push(self.get_localising_moment_matrix(
                        inequality,
                        (2 * level - inequality.degree()) / 2,
                        &monomials_sets,
                        &new_moment_matrix,
                    )?);
                }
            }
            self.localising_moment_matrices_inequalities
                .insert(moment_matrix_id, new_localising_moment_matrices_inequalities);

            self.moment_matrices.insert(moment_matrix_id, new_moment_matrix);
            self.generating_sets.insert(moment_matrix_id, monomials_sets.iter().flatten().cloned().collect());
        }

        info!("Finished setting relaxation.");
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn get_localising_moment_matrix(
        &self,
        polynomial: &Polynomial<Monomial<Data>, Scalar>,
        level: u8,
        monomials_sets: &[BTreeSet<Monomial<Data>>],
        moment_matrix: &RustMomentMatrix<Scalar, Monomial<Data>>,
    ) -> PyResult<RustMomentMatrix<Scalar, Monomial<Data>>>
    where
        Monomial<Data>: Display + RewritingTrait<Monomial<Data>>,
        Polynomial<Monomial<Data>, Scalar>: Display,
    {
        let mut new_localising_moment_matrix = RustMomentMatrix {
            data: BTreeMap::new(),
            size: monomials_sets.iter().take((level + 1).into()).map(|set| set.len()).sum(),
        };

        for (index_row, operator_row) in monomials_sets.iter().take((level + 1).into()).flatten().enumerate() {
            // FIXME: using skip is suboptimal, since it still traverses the iterator
            for (index_col, operator_col) in
                monomials_sets.iter().take((level + 1).into()).flatten().enumerate().skip(index_row)
            {
                // FIXME: performance: no need to recompute the adjoint each time
                let operator_row_adjoint = operator_row.adjoint();
                trace!(
                    "Rewriting {} * {} * {}, before inserting it to the localizing matrix.",
                    operator_row_adjoint, polynomial, operator_col
                );
                let intermediate = (operator_row_adjoint * polynomial).map_err(PyValueError::new_err)?;
                let new_polynomial = (intermediate * operator_col)
                    .map_err(PyValueError::new_err)?
                    .rewrite(self.substitution_strategy, &self.substitutions)
                    .map_err(PyValueError::new_err)?;
                trace!("Adding the rewritten polynomial {} to the localizing matrix.", new_polynomial);

                for (monomial, coefficient) in new_polynomial.data {
                    if let Some((position_matrix, position_matrix_conj)) = new_localising_moment_matrix
                        .get_mut(&monomial, self.substitution_strategy, &self.substitutions)
                        .map_err(PyValueError::new_err)?
                    {
                        // Accumulate rather than insert: if `monomial` or its adjoint has already
                        // been processed for this same (row, col), we must add to the existing
                        // coefficient instead of overwriting it.
                        let base = position_matrix.entry((index_row, index_col)).or_insert(Scalar::zero());
                        *base = *base + coefficient;

                        // On the diagonal the mirror position coincides with the base, so writing
                        // it again would double-count.
                        if index_row != index_col {
                            let mirror = if let Some(position_matrix_conj) = position_matrix_conj {
                                position_matrix_conj.entry((index_col, index_row)).or_insert(Scalar::zero())
                            } else {
                                position_matrix.entry((index_col, index_row)).or_insert(Scalar::zero())
                            };
                            *mirror = *mirror + coefficient.conjugate();
                        }
                    } else {
                        // Use the moment matrix's canonical form as the key so that the localising
                        // matrix and the moment matrix agree on which form (monomial vs. adjoint)
                        // identifies each equivalence class.
                        let canonical_info = moment_matrix
                            .get_canonical(&monomial, self.substitution_strategy, &self.substitutions)
                            .ok();
                        let (key, new_entry) = match canonical_info {
                            // Real entry (whether known via the moment matrix or implied by a
                            // real-valued problem): single matrix, both (row, col) and (col, row)
                            // populated.
                            Some((canonical, _, true)) => (
                                canonical,
                                (
                                    BTreeMap::from([
                                        ((index_row, index_col), coefficient),
                                        ((index_col, index_row), coefficient.conjugate()),
                                    ]),
                                    None,
                                ),
                            ),
                            // Complex entry, monomial is canonical
                            Some((canonical, false, false)) => (
                                canonical,
                                (
                                    BTreeMap::from([((index_row, index_col), coefficient)]),
                                    Some(BTreeMap::from([((index_col, index_row), coefficient.conjugate())])),
                                ),
                            ),
                            // Complex entry, monomial is the adjoint of the canonical form
                            Some((canonical, true, false)) => (
                                canonical,
                                (
                                    BTreeMap::from([((index_col, index_row), coefficient.conjugate())]),
                                    Some(BTreeMap::from([((index_row, index_col), coefficient)])),
                                ),
                            ),
                            None => {
                                return Err(PyValueError::new_err(format!(
                                    "Couldn't find the monomial {} in the moment matrix index.",
                                    monomial
                                )));
                            }
                        };
                        new_localising_moment_matrix.data.insert(key, new_entry);
                    }
                }
            }
        }

        Ok(new_localising_moment_matrix)
    }
}
