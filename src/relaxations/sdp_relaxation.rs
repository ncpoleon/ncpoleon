use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::ops::Mul;

use itertools::Itertools;
use kdam::tqdm;
use log::{debug, info, trace, warn};
use num_complex::Complex;
use num_traits::Zero;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyFloat, PyInt, PyList};

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::{
    PythonCommutativeMonomial, RustCommutativeMonomial,
};
use crate::polynomials::commutative_polynomials::operators::commutative_operator::{
    PythonCommutativeOperator, RustCommutativeOperator,
};
use crate::polynomials::commutative_polynomials::polynomials::commutative_polynomial::{
    PythonComplexCoefficientsCommutativePolynomial, PythonRealCoefficientsCommutativePolynomial,
};
use crate::polynomials::monomial::{
    AdjointTrait, HasAMomentMatrixId, HasLength, Monomial, OneWithMomentMatrixId, RewritingStrategy, RewritingTrait,
};
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::{
    PythonNonCommutativeMonomial, RustNonCommutativeMonomial,
};
use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::{
    PythonNonCommutativeOperator, RustNonCommutativeOperator,
};
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
use crate::utils::merge_btreemaps::merge_btreemaps;

/// Macro to convert Python polynomial objects into Rust types, create an SDP relaxation, and wrap
/// it in the appropriate Python class. Parameterized by the Python polynomial type and relaxation
/// wrapper type.
macro_rules! build_relaxation_inner {
    (
        $py:expr, $level:expr, $objective:expr,
        $operator_constraints_with_generating_sets:expr, $moment_constraints_some:expr, $normalization_constraints_some:expr,
        $variables:expr, $extra_monomials:expr, $unique_moment_id:expr, $substitutions:expr, $strategy:expr,
        $py_poly:ident, $py_relaxation:ident, $py_constraint:ident, $verbosity:expr, $check_uniqueness_with_length:expr $(,)?
    ) => {{
        let rust_objective = $py_poly::try_from_reference_bound($objective, $unique_moment_id)?.0;

        let mut rust_equalities = Vec::new();
        let mut rust_inequalities = Vec::new();
        debug!("Converting operator constraints.");
        for (index, (constraint, generating_set)) in $operator_constraints_with_generating_sets.into_iter().enumerate() {
            let constraint = $py_constraint::try_from(&constraint)?.0;
            let kind = constraint.kind;
            let diff = constraint.into_polynomial_diff().map_err(PyValueError::new_err)?;
            match kind {
                ConstraintKind::Equality => {
                    debug!("Adding polynomial at index {} to the equalities. ({})", index, diff);
                    rust_equalities.push((diff, generating_set));
                }
                ConstraintKind::Inequality => {
                    debug!("Adding polynomial at index {} to the inequalities. ({})", index, diff);
                    rust_inequalities.push((diff, generating_set));
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

        let mut relaxation = SdpRelaxation::new($substitutions, $strategy, $extra_monomials).map_err(PyValueError::new_err)?;
        info!("Setting relaxation.");
        relaxation.set_relaxation(
            $level,
            $variables,
            rust_objective,
            rust_equalities,
            rust_inequalities,
            rust_moment_equalities,
            rust_moment_inequalities,
            rust_normalization_equalities,
            rust_normalization_inequalities,
            $verbosity,
            $check_uniqueness_with_length
        )?;
        $py_relaxation(relaxation).into_py_any($py)
    }};
}

/// Macro to handle the full conversion pipeline for a given monomial type: parse substitutions,
/// branch on real vs complex, and call `build_relaxation_inner!`.
macro_rules! build_relaxation_arm {
    (
        $py:expr, $level:expr, $objective:expr,
        $operator_constraints_with_generating_sets:expr, $moment_constraints_some:expr, $normalization_constraints_some:expr,
        $extra_monomials_some: expr, $substitutions_some:expr, $substitution_strategy:expr,
        operators: $py_operator:ident & $rust_operator:ty,
        monomials: $py_monomial:ident & $rust_monomial:ty,
        $unique_moment_id:expr,
        variables: $variables:expr,
        real_poly_and_relaxation: $real_py_poly:ident & $real_py_relaxation:ident & $real_py_constraint:ident,
        complex_poly_and_relaxation: $complex_py_poly:ident & $complex_py_relaxation:ident & $complex_py_constraint:ident,
        is_real: $is_real:expr, verbosity: $verbosity:expr, check_uniqueness_with_length:$check_uniqueness_with_length:expr $(,)?
    ) => {{
        debug!("Converting variables.");
        let mut variables: Vec<$rust_operator> = Vec::with_capacity($variables.len());

        for (index, op) in $variables.into_iter().enumerate() {
            if let Ok(rust_op) = $py_operator::try_from(op) {
                variables.push(rust_op.0);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Couldn't convert variable at index {} to an operator.",
                    index
                )));
            }
        }

        debug!("Converting extra monomials.");
        let mut rust_extra_monomials: BTreeMap<u8, Vec<$rust_monomial>> = BTreeMap::new();

        for (index, monom) in $extra_monomials_some.into_iter().enumerate() {
            if let Ok(rust_monom) = $py_monomial::try_from_reference_bound(&monom, $unique_moment_id) {
                rust_extra_monomials
                    .entry(rust_monom.0.moment_matrix_id())
                    .or_default()
                    .push(rust_monom.0);
            } else {
                return Err(PyValueError::new_err(format!(
                    "Couldn't convert extra monomial at index {} to a monomial.",
                    index
                )));
            }
        }

        let operator_constraints_with_generating_sets: Vec<(Bound<'_, PyAny>, Option<Vec<$rust_monomial>>)> =
            $operator_constraints_with_generating_sets.into_iter().map(|(constraint, generating_set_option)| {
                let generating_set = generating_set_option.map(|generating_set| {
                    generating_set.into_iter().map(|variable| {
                        $py_monomial::try_from_reference_bound(&variable, $unique_moment_id).map(|res| res.0)
                    }).collect::<Result<Vec<$rust_monomial>, PyErr>>()
                }).transpose()?;

                Ok((constraint, generating_set))
            }).collect::<Result<Vec<_>, PyErr>>()?;

        debug!("Converting substitutions.");
        let mut rust_substitutions: BTreeMap<$rust_monomial, $rust_monomial> = BTreeMap::new();

        for (index, (monom_key, monom_value)) in $substitutions_some.into_iter().enumerate() {
            let try_rust_monom_key = $py_monomial::try_from_reference_bound(&monom_key, $unique_moment_id);
            let try_rust_monom_value = $py_monomial::try_from_reference_bound(&monom_value, $unique_moment_id);

            match (try_rust_monom_key, try_rust_monom_value) {
                (Ok(key), Ok(mut value)) => {
                    // If the RHS term is the identity, it may have been converted from 1, in which
                    // case the conversion couldn't know the moment_matrix index. We set it to the
                    // same one as the monomial to replace
                    // TODO: we should instead check in the setup whether there is a unique moment matix identifier. If yes,
                    //  we can convert 1 or 1.0 to the intended monomial, and error otherwise
                    if value.0.is_one() {
                        warn!("Set the moment matrix index of the identity operator to the same one as {} in a substitution constraint.", key.0);
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
            build_relaxation_inner!(
                $py,
                $level,
                $objective,
                operator_constraints_with_generating_sets,
                $moment_constraints_some,
                $normalization_constraints_some,
                variables,
                rust_extra_monomials,
                $unique_moment_id,
                rust_substitutions,
                $substitution_strategy,
                $real_py_poly,
                $real_py_relaxation,
                $real_py_constraint,
                $verbosity,
                $check_uniqueness_with_length
            )
        } else {
            build_relaxation_inner!(
                $py,
                $level,
                $objective,
                operator_constraints_with_generating_sets,
                $moment_constraints_some,
                $normalization_constraints_some,
                variables,
                rust_extra_monomials,
                $unique_moment_id,
                rust_substitutions,
                $substitution_strategy,
                $complex_py_poly,
                $complex_py_relaxation,
                $complex_py_constraint,
                $verbosity,
                $check_uniqueness_with_length
            )
        }
    }};
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
///   whether it is the case by trying to convert every polynomial to a real-valued one. Set this argument to `True` to
///   speed up the initial step of the relaxation if you know that your problem is real-valued.
/// * `assume_complex` – If `True`, the function assumes that the problem is complex-valued, instead of trying to infer
///   whether it is the case by trying to convert every polynomial to a real-valued one. Set this argument to `True` to
///   speed up the initial step of the relaxation if you know that your problem is complex-valued.
/// * `assume_commutative` – If `True`, the function assumes that the problem uses only commutative variables, instead
///   of trying to infer whether it is the case by trying to convert every polynomial to a commutative one. Set this
///   argument to `True` to speed up the initial step of the relaxation if your problem uses commutative variables only.
/// * `assume_noncommutative` – If `True`, the function assumes that the problem uses only noncommutative variables,
///   instead of trying to infer whether it is the case by trying to convert every polynomial to a noncommutative one.
///   Set this argument to `True` to speed up the initial step of the relaxation if your problem uses commutative
///   variables only.
/// * `extra_monomials` – Extra monomials to be added to the generating set of the moment matrix. They're not taken into
///   account for the localizing matrices.
/// * `verbosity` – The level of verbosity of the relaxation. Notably, it controls whether progress bars are printed.
/// * `check_uniqueness_with_length` – If `True`, then it is assumed that a monomial that is rewritten can always be
///   expressed as a product of the operators present in `variables` and that rewriting a monomial can't increase its
///   length. This allows to check more quickly whether a given monomial should be kept in the indexing set of the
///   moment matrix.
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
        extra_monomials=None,
        verbosity=0,
        check_uniqueness_with_length=true,
    )
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_relaxation<'py>(
    variables: &Bound<'py, PyList>,
    level: i8,
    objective: &Bound<'py, PyAny>,
    substitutions: Option<&Bound<'py, PyDict>>,
    operator_constraints: Option<&Bound<'py, PyList>>,
    moment_constraints: Option<&Bound<'py, PyList>>,
    normalization_constraints: Option<&Bound<'py, PyList>>,
    substitution_strategy: RewritingStrategy,
    extra_monomials: Option<&Bound<'py, PyList>>,
    verbosity: u8,
    check_uniqueness_with_length: bool,
) -> PyResult<Py<PyAny>> {
    let py = objective.py();
    let default_dict = PyDict::new(py);
    let default_list = PyList::empty(py);
    let substitutions_some = substitutions.unwrap_or(&default_dict);
    let operator_constraints_some = operator_constraints.unwrap_or(&default_list);
    let moment_constraints_some = moment_constraints.unwrap_or(&default_list);
    let normalization_constraints_some = normalization_constraints.unwrap_or(&default_list);
    let extra_monomials_some = extra_monomials.unwrap_or(&default_list);

    // We first need to check whether all the objects are real or complex-valued and their commutativity type
    let mut is_problem_real_valued = true;
    let mut problem_contains_commutative: bool = false;
    let mut problem_contains_noncommutative: bool = false;

    let (realness, commutativity, mut unique_moment_id) =
        get_realness_and_commutativity_and_unique_mm_id_of_polynomial_from_bound(objective)
            .map_err(|_err| PyValueError::new_err("Couldn't convert the objective into a supported polynomial."))?;

    is_problem_real_valued &= realness;

    if let Some(commutativity_type) = commutativity {
        match commutativity_type {
            MonomialCommutativity::Commutative => problem_contains_commutative = true,
            MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
        }
    }

    for (label, constraints_list) in
        [("moment", moment_constraints_some), ("normalization", normalization_constraints_some)]
    {
        for (index, value) in constraints_list.into_iter().enumerate() {
            let (realness, commutativity) =
                get_realness_and_commutativity_of_constraint_from_bound(&value).map_err(|_err| {
                    PyValueError::new_err(format!(
                        "Couldn't convert the {} constraint at index {} to a supported constraint.",
                        label, index
                    ))
                })?;

            is_problem_real_valued &= realness;

            if let Some(commutativity_type) = commutativity {
                match commutativity_type {
                    MonomialCommutativity::Commutative => problem_contains_commutative = true,
                    MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
                }
            }
        }
    }

    let mut operator_constraints_with_generating_sets = Vec::with_capacity(operator_constraints_some.len());

    for (index, constraint) in operator_constraints_some.into_iter().enumerate() {
        let (constraint, generating_set_some) = constraint
            .extract::<(Bound<'_, PyAny>, Option<Bound<'_, PyList>>)>()
            .unwrap_or_else(|_err| (constraint, None));

        if let Some(generating_set) = &generating_set_some {
            for (index_variable, variable) in generating_set.iter().enumerate() {
                let (commutativity, mm_id) = get_commutativity_and_mm_index_from_bound(&variable).map_err(|_err| {
                    PyValueError::new_err(format!(
                        "Couldn't convert the variable at index {} in the operator constraint's generating set at
                                index {} to a supported monomial.",
                        index_variable, index
                    ))
                })?;

                if let Some(commutativity_type) = commutativity {
                    match commutativity_type {
                        MonomialCommutativity::Commutative => problem_contains_commutative = true,
                        MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
                    }
                }

                unique_moment_id = match (unique_moment_id, mm_id) {
                    (Some(mm_id1), Some(mm_id2)) => {
                        if mm_id1 == mm_id2 {
                            Some(mm_id1)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
            }
        }

        let (realness, commutativity) =
            get_realness_and_commutativity_of_constraint_from_bound(&constraint).map_err(|_err| {
                PyValueError::new_err(format!(
                    "Couldn't convert the operator constraint at index {} to a supported constraint.",
                    index
                ))
            })?;

        is_problem_real_valued &= realness;

        if let Some(commutativity_type) = commutativity {
            match commutativity_type {
                MonomialCommutativity::Commutative => problem_contains_commutative = true,
                MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
            }
        }

        operator_constraints_with_generating_sets.push((constraint, generating_set_some));
    }

    for (variables_set_label, variables_set) in [("variable", variables), ("extra monomial", extra_monomials_some)] {
        for (index_variable, variable) in variables_set.into_iter().enumerate() {
            let (commutativity, mm_id) = get_commutativity_and_mm_index_from_bound(&variable).map_err(|_err| {
                PyValueError::new_err(format!(
                    "Couldn't convert the {} at index {} to a supported monomial.",
                    variables_set_label, index_variable
                ))
            })?;

            if let Some(commutativity_type) = commutativity {
                match commutativity_type {
                    MonomialCommutativity::Commutative => problem_contains_commutative = true,
                    MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
                }
            }

            unique_moment_id = match (unique_moment_id, mm_id) {
                (Some(mm_id1), Some(mm_id2)) => {
                    if mm_id1 == mm_id2 {
                        Some(mm_id1)
                    } else {
                        None
                    }
                }
                _ => None,
            };
        }
    }

    for (index, (key, value)) in substitutions_some.into_iter().enumerate() {
        // TODO: this should be moved to using get_realness_and_commutativity_of_polynomial_from_bound when supporting
        // polynomials for substitutions
        let (commutativity, mm_id) = get_commutativity_and_mm_index_from_bound(&key).map_err(|_err| {
            PyValueError::new_err(format!(
                "Couldn't convert the key at index {} of the substitutions to a supported monomial.",
                index
            ))
        })?;

        if let Some(commutativity_type) = commutativity {
            match commutativity_type {
                MonomialCommutativity::Commutative => problem_contains_commutative = true,
                MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
            }
        }

        unique_moment_id = match (unique_moment_id, mm_id) {
            (Some(mm_id1), Some(mm_id2)) => {
                if mm_id1 == mm_id2 {
                    Some(mm_id1)
                } else {
                    None
                }
            }
            _ => None,
        };

        let (commutativity, mm_id) = get_commutativity_and_mm_index_from_bound(&value).map_err(|_err| {
            PyValueError::new_err(format!(
                "Couldn't convert the substitute at index {} of the substitutions to a supported monomial.",
                index
            ))
        })?;

        if let Some(commutativity_type) = commutativity {
            match commutativity_type {
                MonomialCommutativity::Commutative => problem_contains_commutative = true,
                MonomialCommutativity::NonCommutative => problem_contains_noncommutative = true,
            }
        }

        unique_moment_id = match (unique_moment_id, mm_id) {
            (Some(mm_id1), Some(mm_id2)) => {
                if mm_id1 == mm_id2 {
                    Some(mm_id1)
                } else {
                    None
                }
            }
            _ => None,
        };
    }

    match (problem_contains_commutative, problem_contains_noncommutative) {
        (false, false) => Err(PyValueError::new_err("Variables must be provided.")),
        // Noncommutative problem
        (false, true) => {
            build_relaxation_arm!(
                py, level, objective,
                operator_constraints_with_generating_sets, moment_constraints_some, normalization_constraints_some,
                extra_monomials_some, substitutions_some, substitution_strategy,
                operators: PythonNonCommutativeOperator & RustNonCommutativeOperator,
                monomials: PythonNonCommutativeMonomial & RustNonCommutativeMonomial,
                unique_moment_id,
                variables: variables,
                real_poly_and_relaxation: PythonRealCoefficientsNonCommutativePolynomial &
                    PythonRealValuedNonCommutativeSdpRelaxation &
                    PythonRealCoefficientsNonCommutativeConstraint,
                complex_poly_and_relaxation: PythonComplexCoefficientsNonCommutativePolynomial &
                    PythonComplexValuedNonCommutativeSdpRelaxation &
                    PythonComplexCoefficientsNonCommutativeConstraint,
                is_real: is_problem_real_valued, verbosity: verbosity, check_uniqueness_with_length: check_uniqueness_with_length
            )
        }
        // Commutative problem
        (true, false) => {
            build_relaxation_arm!(
                py, level, objective,
                operator_constraints_with_generating_sets, moment_constraints_some, normalization_constraints_some,
                extra_monomials_some, substitutions_some, substitution_strategy,
                operators: PythonCommutativeOperator & RustCommutativeOperator,
                monomials: PythonCommutativeMonomial & RustCommutativeMonomial,
                unique_moment_id,
                variables: variables,
                real_poly_and_relaxation: PythonRealCoefficientsCommutativePolynomial &
                    PythonRealValuedCommutativeSdpRelaxation &
                    PythonRealCoefficientsCommutativeConstraint,
                complex_poly_and_relaxation: PythonComplexCoefficientsCommutativePolynomial &
                    PythonComplexValuedCommutativeSdpRelaxation &
                    PythonComplexCoefficientsCommutativeConstraint,
                is_real: is_problem_real_valued, verbosity: verbosity, check_uniqueness_with_length: check_uniqueness_with_length
            )
        }
        (true, true) => Err(PyNotImplementedError::new_err(
            "Hybrid polynomials are not handled yet, but both commutative and \
                non-commutative operators have been detected.",
        )),
    }
}

// Having an enum will be simpler than a bool when we'll introduce Hybrid monomials
enum MonomialCommutativity {
    Commutative,
    NonCommutative,
}

/// Get the commutativity of an operator or a monomial
fn get_commutativity_and_mm_index_from_bound<'py>(
    bound: &Bound<'py, PyAny>,
) -> Result<(Option<MonomialCommutativity>, Option<u8>), ()> {
    if bound.cast::<PyInt>().is_ok() || bound.cast::<PyFloat>().is_ok() || bound.cast::<PyComplex>().is_ok() {
        Ok((None, None))
    } else if let Ok(op_bound) = bound.cast::<PythonCommutativeOperator>() {
        Ok((Some(MonomialCommutativity::Commutative), Some(op_bound.get().0.moment_matrix_id())))
    } else if let Ok(monom_bound) = bound.cast::<PythonCommutativeMonomial>() {
        Ok((Some(MonomialCommutativity::Commutative), Some(monom_bound.get().0.moment_matrix_id())))
    } else if let Ok(op_bound) = bound.cast::<PythonNonCommutativeOperator>() {
        Ok((Some(MonomialCommutativity::NonCommutative), Some(op_bound.get().0.moment_matrix_id())))
    } else if let Ok(monom_bound) = bound.cast::<PythonNonCommutativeMonomial>() {
        Ok((Some(MonomialCommutativity::NonCommutative), Some(monom_bound.get().0.moment_matrix_id())))
    } else {
        Err(())
    }
}

/// Get whether a polynomial a real-valued and its type of variables
fn get_realness_and_commutativity_and_unique_mm_id_of_polynomial_from_bound<'py>(
    bound: &Bound<'py, PyAny>,
) -> Result<(bool, Option<MonomialCommutativity>, Option<u8>), ()> {
    if bound.cast::<PyInt>().is_ok() || bound.cast::<PyFloat>().is_ok() {
        Ok((true, None, None))
    } else if bound.cast::<PyComplex>().is_ok() {
        Ok((false, None, None))
    } else if let Ok(op_bound) = bound.cast::<PythonCommutativeOperator>() {
        Ok((true, Some(MonomialCommutativity::Commutative), Some(op_bound.get().0.moment_matrix_id())))
    } else if let Ok(monom_bound) = bound.cast::<PythonCommutativeMonomial>() {
        Ok((true, Some(MonomialCommutativity::Commutative), Some(monom_bound.get().0.moment_matrix_id())))
    } else if let Ok(op_bound) = bound.cast::<PythonNonCommutativeOperator>() {
        Ok((true, Some(MonomialCommutativity::NonCommutative), Some(op_bound.get().0.moment_matrix_id())))
    } else if let Ok(monom_bound) = bound.cast::<PythonNonCommutativeMonomial>() {
        Ok((true, Some(MonomialCommutativity::NonCommutative), Some(monom_bound.get().0.moment_matrix_id())))
    } else if let Ok(poly_bound) = bound.cast::<PythonRealCoefficientsCommutativePolynomial>() {
        Ok((true, Some(MonomialCommutativity::Commutative), poly_bound.get().0.get_unique_moment_matrix_id()))
    } else if let Ok(poly_bound) = bound.cast::<PythonRealCoefficientsNonCommutativePolynomial>() {
        Ok((true, Some(MonomialCommutativity::NonCommutative), poly_bound.get().0.get_unique_moment_matrix_id()))
    } else if let Ok(poly_bound) = bound.cast::<PythonComplexCoefficientsCommutativePolynomial>() {
        Ok((false, Some(MonomialCommutativity::Commutative), poly_bound.get().0.get_unique_moment_matrix_id()))
    } else if let Ok(poly_bound) = bound.cast::<PythonComplexCoefficientsNonCommutativePolynomial>() {
        Ok((false, Some(MonomialCommutativity::NonCommutative), poly_bound.get().0.get_unique_moment_matrix_id()))
    } else {
        Err(())
    }
}

fn get_realness_and_commutativity_of_constraint_from_bound<'py>(
    bound: &Bound<'py, PyAny>,
) -> Result<(bool, Option<MonomialCommutativity>), ()> {
    if bound.cast::<PythonRealCoefficientsCommutativeConstraint>().is_ok() {
        Ok((true, Some(MonomialCommutativity::Commutative)))
    } else if bound.cast::<PythonComplexCoefficientsCommutativeConstraint>().is_ok() {
        Ok((false, Some(MonomialCommutativity::Commutative)))
    } else if bound.cast::<PythonRealCoefficientsNonCommutativeConstraint>().is_ok() {
        Ok((true, Some(MonomialCommutativity::NonCommutative)))
    } else if bound.cast::<PythonComplexCoefficientsNonCommutativeConstraint>().is_ok() {
        Ok((false, Some(MonomialCommutativity::NonCommutative)))
    } else {
        Err(())
    }
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
                let unique_moment_matrix_id = if self.0.moment_matrices.keys().len() == 1 {
                    Some(*self.0.moment_matrices.first_key_value().unwrap().0)
                } else {
                    None
                };
                if let Ok(mon) = $py_monomial::try_from_reference_bound(mon_or_poly, unique_moment_matrix_id) {
                    $py_monomial(
                        mon.0
                        .rewrite(
                            self.0.substitution_strategy,
                            &self.0.substitutions
                        )
                        .map_err(PyValueError::new_err)?
                    ).into_py_any(py)
                } else {
                    let poly = $py_poly::try_from_reference_bound(mon_or_poly, unique_moment_matrix_id)?;
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

            /// Dictionary of all generating sets including extra monomials
            ///
            /// Each element corresponds to a unique moment matrix identifier.
            #[getter]
            fn generating_sets(&self) -> BTreeMap<u8, Vec<$py_monomial>> {
                let left_btreemap: BTreeMap<u8, Vec<$py_monomial>> = self.0
                    .generating_sets
                    .iter()
                    .map(|(&index, generating_set)| (index, generating_set.iter().cloned().map($py_monomial).collect())).collect();
                let right_btreemap: BTreeMap<u8, Vec<$py_monomial>> = self.0
                    .extra_monomials
                    .iter()
                    .map(|(&index, generating_set)| (index, generating_set.iter().cloned().map($py_monomial).collect())).collect();
                merge_btreemaps(
                    left_btreemap,
                    right_btreemap,
                    |_, vec_left, vec_right| [vec_left, vec_right].concat()
                )
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
            fn equalities(&self) -> BTreeMap<u8, Vec<($py_poly, Vec<$py_monomial>)>> {
                self.0
                    .equalities
                    .iter()
                    .map(|(&mm_id, equalities_id)| {
                        (
                            mm_id,
                            equalities_id
                                .iter()
                                .map(|(poly, generating_set)| {
                                    (
                                        $py_poly(poly.clone()),
                                        generating_set
                                            .iter()
                                            .map(|rust_monomial| $py_monomial(rust_monomial.clone()))
                                            .collect(),
                                    )
                                })
                                .collect(),
                        )
                    })
                    .collect()
            }

            #[getter]
            fn inequalities(&self) -> BTreeMap<u8, Vec<($py_poly, Vec<$py_monomial>)>> {
                self.0
                    .inequalities
                    .iter()
                    .map(|(&mm_id, inequalities_id)| {
                        (
                            mm_id,
                            inequalities_id
                                .iter()
                                .map(|(poly, generating_set)| {
                                    (
                                        $py_poly(poly.clone()),
                                        generating_set
                                            .iter()
                                            .map(|rust_monomial| $py_monomial(rust_monomial.clone()))
                                            .collect(),
                                    )
                                })
                                .collect(),
                        )
                    })
                    .collect()
            }
        }
    };
}

type PolynomialWithGeneratingSet<MonomialType, Scalar> = (Polynomial<MonomialType, Scalar>, Vec<MonomialType>);

pub(super) struct SdpRelaxation<MonomialType: AdjointTrait + Ord, Scalar: PolynomialDtype> {
    objective: Polynomial<MonomialType, Scalar>,
    substitutions: BTreeMap<MonomialType, MonomialType>,
    substitution_strategy: RewritingStrategy,
    equalities: BTreeMap<u8, Vec<PolynomialWithGeneratingSet<MonomialType, Scalar>>>,
    inequalities: BTreeMap<u8, Vec<PolynomialWithGeneratingSet<MonomialType, Scalar>>>,
    moment_equalities: Vec<(Polynomial<MonomialType, Scalar>, Scalar)>,
    moment_inequalities: Vec<(Polynomial<MonomialType, Scalar>, f64)>,
    moment_matrices: BTreeMap<u8, RustMomentMatrix<Scalar, MonomialType>>,
    generating_sets: BTreeMap<u8, Vec<MonomialType>>,
    localising_moment_matrices_equalities: BTreeMap<u8, Vec<RustMomentMatrix<Scalar, MonomialType>>>,
    localising_moment_matrices_inequalities: BTreeMap<u8, Vec<RustMomentMatrix<Scalar, MonomialType>>>,
    extra_monomials: BTreeMap<u8, Vec<MonomialType>>,
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

type PolynomialWithOptionalGeneratingSet<MonomialType, Scalar> =
    (Polynomial<MonomialType, Scalar>, Option<Vec<MonomialType>>);

impl<Data: Ord + Clone, Scalar: PolynomialDtype> SdpRelaxation<Monomial<Data>, Scalar>
where
    Polynomial<Monomial<Data>, Scalar>: PolynomialTrait,
    Monomial<Data>: RewritingTrait<Monomial<Data>> + OneWithMomentMatrixId + AdjointTrait,
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    for<'a> Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    for<'a> Monomial<Data>:
        Mul<&'a Polynomial<Monomial<Data>, Scalar>, Output = Result<Polynomial<Monomial<Data>, Scalar>, String>>,
    for<'a> Polynomial<Monomial<Data>, Scalar>:
        Mul<&'a Monomial<Data>, Output = Result<Polynomial<Monomial<Data>, Scalar>, String>>,
{
    pub(super) fn new(
        substitutions: BTreeMap<Monomial<Data>, Monomial<Data>>,
        substitution_strategy: RewritingStrategy,
        extra_monomials: BTreeMap<u8, Vec<Monomial<Data>>>,
    ) -> Result<Self, String> {
        let rewritten_extra_monomials: BTreeMap<u8, Vec<Monomial<Data>>> = extra_monomials
            .into_iter()
            .map(|(moment_matrix_index, monomials)| {
                monomials
                    .into_iter()
                    .map(|monomial| monomial.rewrite(substitution_strategy, &substitutions))
                    .collect::<Result<Vec<Monomial<Data>>, String>>()
                    .map(|rewritten_monomials| (moment_matrix_index, rewritten_monomials))
            })
            .collect::<Result<BTreeMap<u8, Vec<Monomial<Data>>>, String>>()?;

        Ok(Self {
            objective: Polynomial::zero(),
            substitutions,
            substitution_strategy,
            equalities: BTreeMap::new(),
            inequalities: BTreeMap::new(),
            moment_equalities: Vec::with_capacity(0),
            moment_inequalities: Vec::with_capacity(0),
            moment_matrices: BTreeMap::new(),
            generating_sets: BTreeMap::new(),
            localising_moment_matrices_equalities: BTreeMap::new(),
            localising_moment_matrices_inequalities: BTreeMap::new(),
            extra_monomials: rewritten_extra_monomials,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn set_relaxation<OperatorType: Copy + Ord + AdjointTrait + Display + HasAMomentMatrixId>(
        &mut self,
        level: i8,
        variables: Vec<OperatorType>,
        objective: Polynomial<Monomial<Data>, Scalar>,
        equalities: Vec<PolynomialWithOptionalGeneratingSet<Monomial<Data>, Scalar>>,
        inequalities: Vec<PolynomialWithOptionalGeneratingSet<Monomial<Data>, Scalar>>,
        moment_equalities: Vec<(Polynomial<Monomial<Data>, Scalar>, Scalar)>,
        moment_inequalities: Vec<(Polynomial<Monomial<Data>, Scalar>, f64)>,
        normalization_equalities: Vec<(Polynomial<Monomial<Data>, Scalar>, Scalar)>,
        normalization_inequalities: Vec<(Polynomial<Monomial<Data>, Scalar>, f64)>,
        verbosity: u8,
        check_uniqueness_with_length: bool,
    ) -> PyResult<()>
    where
        Monomial<Data>: RewritingTrait<Monomial<Data>> + Display + HasLength,
        for<'a, 'b> &'a Monomial<Data>: Mul<&'b OperatorType, Output = Result<Monomial<Data>, String>>,
        Polynomial<Monomial<Data>, Scalar>: RewritingTrait<Monomial<Data>> + Display,
    {
        if level < -1 {
            return Err(PyValueError::new_err(format!(
                "level must be larger than or equal to -1 but {} was given.",
                level
            )));
        }

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

        let all_moment_matrices_identifers =
            BTreeSet::from_iter(variables_with_adjoint.keys().copied().chain(self.extra_monomials.keys().copied()));

        // We do this here so that if there's an error, the user doesn't have to wait for the
        // generating sets to be created. The temporary BTreeMaps are here so that self isn't
        // borrowed mutably when we change the generating sets for these constraints
        let mut temporary_equalities: BTreeMap<u8, Vec<PolynomialWithGeneratingSet<Monomial<Data>, Scalar>>> =
            BTreeMap::new();
        let mut temporary_inequalities: BTreeMap<u8, Vec<PolynomialWithGeneratingSet<Monomial<Data>, Scalar>>> =
            BTreeMap::new();
        macro_rules! partition_operator_constraints {
            ($constraints_field:ident, $temporary:expr, $constraint_kind:literal) => {{
                debug!("Partitioning operator {} constraints.", stringify!($constraints_field));
                for (index, (constraint, generating_set)) in $constraints_field.into_iter().enumerate() {
                    if let Some(moment_matrix_id) = constraint.get_unique_moment_matrix_id() {
                        if !all_moment_matrices_identifers.contains(&moment_matrix_id) {
                            return Err(PyValueError::new_err(format!(
                                "The polynomial at index {} in the operator {} constraints is defined using the moment
                                matrix identifier {} which isn't associated with a moment matrix.",
                                index, $constraint_kind, moment_matrix_id
                            )));
                        }
                        $temporary.entry(moment_matrix_id).or_default().push((
                            constraint
                                .rewrite(self.substitution_strategy, &self.substitutions)
                                .map_err(PyValueError::new_err)?,
                            generating_set.unwrap_or_default(),
                        ));
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "The polynomial at index {} in the operator {} constraints isn't defined using a unique
                            moment matrix identifier.",
                            index, $constraint_kind
                        )));
                    }
                }
            }};
        }

        partition_operator_constraints!(equalities, temporary_equalities, "equality");
        partition_operator_constraints!(inequalities, temporary_inequalities, "inequality");

        // Auto-inject default normalization `<I_k> = 1` for each moment-matrix index `k` that
        // doesn't already appear in a user-supplied normalization constraint. Only normalization
        // constraints contribute to the "covered" set ; generic moment constraints don't, so a user
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
        for &k in all_moment_matrices_identifers.iter() {
            if !covered_indices.contains(&k) && (level > -1) {
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

        let top_bar = (verbosity > 0) && (verbosity < 3) && (variables_with_adjoint.len() > 1);

        let mm_iterator = if top_bar {
            itertools::Either::Left(tqdm!(
                all_moment_matrices_identifers.into_iter(),
                desc = "Moment matrix index",
                position = 0,
                ncols = 0
            ))
        } else {
            itertools::Either::Right(all_moment_matrices_identifers.into_iter())
        };

        for moment_matrix_id in mm_iterator {
            let variables_set = variables_with_adjoint.remove(&moment_matrix_id).unwrap_or_default();
            // The i-th element of monomials_sets contains the set of monomials of length i + 1
            // This allows us to access the monomials for lower k_i when dealing with
            // localizing moment matrices
            let mut monomials_sets = Vec::with_capacity((1 + level) as usize);
            let extra_monomials: &[Monomial<Data>] =
                self.extra_monomials.get(&moment_matrix_id).map_or(&[], Vec::as_slice);

            if level >= 0 {
                monomials_sets.push(BTreeSet::from([Monomial::one(moment_matrix_id)]));
            }

            // Generating the monomials set by finding which monomials can be reduced
            // FIXME: if the monomials are commutative, we can instead loop over the possible powers of
            // the operators at a given level, it's way more efficient (comb(d+t-1, t) vs d**t). In
            // order to do so, we could add a is_commutative method to PolynomialTrait, just like we did
            // with is_real. This however wouldn't work to generate Hybrid monomials, we may want to have
            // two different sets of variables, one commutative and one non commutative
            let positive_level = level.max(0) as u8;
            let monomial_length_iterator = if (verbosity > 0) && (level > 0) {
                itertools::Either::Left(tqdm!(
                    1..=positive_level,
                    desc = "Generating monomials with length",
                    position = if top_bar { 1 } else { 0 },
                    ncols = 0
                ))
            } else {
                // Empty if level = -1 or 0
                itertools::Either::Right(1..=positive_level)
            };

            for monomial_length in monomial_length_iterator {
                let mut level_set = BTreeSet::new();
                if let Some(last_level_set) = monomials_sets.last() {
                    let mut cartesian_product_iterator = if (verbosity > 0) && (verbosity < 3) {
                        itertools::Either::Left(tqdm!(
                            last_level_set.iter().cartesian_product(variables_set.iter()),
                            desc = "Monomial combinations",
                            position = if top_bar { 2 } else { 1 },
                            total = last_level_set.len() * variables_set.len(),
                            leave = false
                        ))
                    } else {
                        itertools::Either::Right(last_level_set.iter().cartesian_product(variables_set.iter()))
                    };

                    cartesian_product_iterator
                        .try_for_each(|(monomial, variable)| -> Result<(), String> {
                            let new_monomial = (monomial * variable)?;
                            // We remove from the monomials set all monomials that can be reduced via
                            // substitutions
                            trace!("New monomial: {}.", new_monomial);
                            let rewritten = new_monomial.rewrite(self.substitution_strategy, &self.substitutions)?;
                            trace!("Rewritten monomial: {}.", rewritten);
                            // We have to check that a reduced monomial has not been inserted in a previous
                            // level. In all generality, we can't simply check that its length is equal to
                            // the current level, since this wouldn't work if the reduced monomial can't be
                            // expressed as a product of the variables that were provided. Furthermore, this
                            // assumes that rewriting a monomial can't increase its length. Though this is
                            // reasonable, we allow the user to disable this simpler check if one of this
                            // assumptions isn't verified
                            if check_uniqueness_with_length {
                                if (rewritten.len() == monomial_length) & !level_set.contains(&rewritten) {
                                    trace!(
                                        "Adding the rewritten monomial to the indexing set at level {}.",
                                        monomial_length
                                    );
                                    level_set.insert(rewritten.clone());
                                }
                            } else {
                                if !level_set.contains(&rewritten)
                                    & !monomials_sets.iter().any(|monomial_set| monomial_set.contains(&rewritten))
                                {
                                    trace!(
                                        "Adding the rewritten monomial to the indexing set at level {}.",
                                        monomial_length
                                    );
                                    level_set.insert(rewritten.clone());
                                }
                            }
                            Ok(())
                        })
                        .map_err(PyValueError::new_err)?;
                    monomials_sets.push(level_set);
                }
            }

            let is_problem_real_valued = self.objective.is_real();
            let mut new_moment_matrix = RustMomentMatrix::new(
                moment_matrix_id, monomials_sets.iter().map(|set| set.len()).sum::<usize>() + extra_monomials.len(),
            );

            // Determine the constraints on the moment matrix. This is where we build the map between
            // reduced monomials and indices within the moment matrix
            let monomials_sets_iterator_rows = if verbosity > 0 {
                itertools::Either::Left(tqdm!(
                    monomials_sets.iter().flatten().chain(extra_monomials.iter()).enumerate(),
                    desc = "Filling moment matrix rows",
                    position = if top_bar { 1 } else { 0 },
                    total = monomials_sets.iter().map(|set| set.len()).sum::<usize>() + extra_monomials.len()
                ))
            } else {
                itertools::Either::Right(monomials_sets.iter().flatten().chain(extra_monomials.iter()).enumerate())
            };

            for (index_row, monomial_row) in monomials_sets_iterator_rows {
                // Computed once per row instead of once per cell.
                let monomial_row_adjoint = monomial_row.adjoint();
                // FIXME: using skip makes it run in n^2 instead of n*(n+1)/2. We can probably fix it
                // by computing how many elements (i.e. lengths) should we skip, and then skip the first
                // remaining elements of the first length that we consider. Maybe write this as a function
                let monomials_sets_iterator_cols =
                    monomials_sets.iter().flatten().chain(extra_monomials.iter()).enumerate().skip(index_row);

                for (index_column, monomial_column) in monomials_sets_iterator_cols {
                    let new_monomial = if index_row == 0 {
                        monomial_column.clone()
                    } else {
                        (&monomial_row_adjoint * monomial_column)
                            .map_err(PyValueError::new_err)?
                            .rewrite(self.substitution_strategy, &self.substitutions)
                            .map_err(PyValueError::new_err)?
                    };

                    // `get_mut` finds the entry (in either orientation) via the matrix's own
                    // `adjoint_index`, so no per-cell `adjoint().rewrite()` is needed here.
                    if let Some((position_matrix, position_matrix_conj)) = new_moment_matrix.get_mut(&new_monomial) {
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
                    // `insert` records the entry and registers its adjoint's canonical form in
                    // `adjoint_index` (the single adjoint rewrite per stored monomial).
                    new_moment_matrix
                        .insert(new_monomial, new_entry, self.substitution_strategy, &self.substitutions)
                        .map_err(PyValueError::new_err)?;
                }
            }

            // The constraints are taken out of the temporary map rather than borrowed from `self`,
            // since filling their generating sets in place would otherwise borrow `self` mutably,
            // which would conflict with the immutable borrow taken by `get_localising_moment_matrix`
            macro_rules! build_localising_moment_matrices {
                ($temporary_constraints:expr, $constraints_field:ident, $matrices_field:ident) => {{
                    let mut new_localising_moment_matrices = Vec::new();
                    if let Some(mut constraints) = $temporary_constraints.remove(&moment_matrix_id) {
                        new_localising_moment_matrices.reserve_exact(constraints.len());
                        let constraints_with_generating_sets_iterator = if verbosity > 0 {
                            itertools::Either::Left(tqdm!(
                                constraints.iter_mut(),
                                desc =
                                    format!("Building localising moment matrices ({})", stringify!($constraints_field)),
                                position = if top_bar { 1 } else { 0 },
                                ncols = 0
                            ))
                        } else {
                            itertools::Either::Right(constraints.iter_mut())
                        };

                        for (constraint, generating_set) in constraints_with_generating_sets_iterator {
                            // No user-provided generating set for this constraint, so we fall back to the
                            // classical level-oriented generating set
                            if generating_set.is_empty() {
                                generating_set.extend(
                                    monomials_sets
                                        .iter()
                                        .take((((2 * level - constraint.degree() as i8) / 2) + 1).max(0) as usize)
                                        .flatten()
                                        .cloned(),
                                );

                                if generating_set.is_empty() {
                                    return Err(PyValueError::new_err(format!(
                                        "({} constraints) Level isn't large enough to deal with the constraint {}. \
                                        Please provide a generating set for this constraint.",
                                        stringify!($constraints_field),
                                        constraint,
                                    )));
                                }
                            }
                            new_localising_moment_matrices.push(self.get_localising_moment_matrix(
                                constraint,
                                generating_set,
                                &new_moment_matrix,
                                verbosity,
                                top_bar,
                            )?);
                        }
                        // The generating sets are now resolved, so the constraints can be stored
                        self.$constraints_field.insert(moment_matrix_id, constraints);
                    }
                    self.$matrices_field.insert(moment_matrix_id, new_localising_moment_matrices);
                }};
            }

            build_localising_moment_matrices!(temporary_equalities, equalities, localising_moment_matrices_equalities);
            build_localising_moment_matrices!(
                temporary_inequalities,
                inequalities,
                localising_moment_matrices_inequalities
            );

            self.moment_matrices.insert(moment_matrix_id, new_moment_matrix);
            self.generating_sets.insert(moment_matrix_id, monomials_sets.iter().flatten().cloned().collect());
        }

        info!("Finished setting relaxation.");
        Ok(())
    }

    fn get_localising_moment_matrix(
        &self,
        polynomial: &Polynomial<Monomial<Data>, Scalar>,
        generating_set: &[Monomial<Data>],
        moment_matrix: &RustMomentMatrix<Scalar, Monomial<Data>>,
        verbosity: u8,
        top_bar: bool,
    ) -> PyResult<RustMomentMatrix<Scalar, Monomial<Data>>>
    where
        Monomial<Data>: Display + RewritingTrait<Monomial<Data>>,
        Polynomial<Monomial<Data>, Scalar>: Display,
    {
        let size = generating_set.len();

        let mut new_localising_moment_matrix = RustMomentMatrix::new(moment_matrix.associated_id, size);

        let monomials_iterator_rows = if verbosity > 0 {
            itertools::Either::Left(tqdm!(
                generating_set.iter().enumerate(),
                desc = "Filling localising matrix rows",
                position = if top_bar { 2 } else { 1 },
                leave = false,
                total = size
            ))
        } else {
            itertools::Either::Right(generating_set.iter().enumerate())
        };

        for (index_row, operator_row) in monomials_iterator_rows {
            // Slicing rather than `skip` keeps this at n*(n+1)/2 instead of n^2.
            let monomials_iterator_cols =
                generating_set[index_row..].iter().enumerate().map(|(offset, operator)| (index_row + offset, operator));

            for (index_col, operator_col) in monomials_iterator_cols {
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
                    if let Some((position_matrix, position_matrix_conj)) =
                        new_localising_moment_matrix.get_mut(&monomial)
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
                        new_localising_moment_matrix
                            .insert(key, new_entry, self.substitution_strategy, &self.substitutions)
                            .map_err(PyValueError::new_err)?;
                    }
                }
            }
        }

        Ok(new_localising_moment_matrix)
    }
}
