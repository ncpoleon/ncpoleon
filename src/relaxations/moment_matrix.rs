use std::collections::BTreeMap;
use std::fmt::Display;

use num_complex::Complex;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use crate::polynomials::commutative_polynomials::monomials::commutative_monomial::{
    PythonCommutativeMonomial, RustCommutativeMonomial,
};
use crate::polynomials::monomial::{AdjointTrait, RewritingStrategy, RewritingTrait};
use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::{
    PythonNonCommutativeMonomial, RustNonCommutativeMonomial,
};
use crate::polynomials::polynomial::PolynomialDtype;

type PositionMatrix<Scalar> = BTreeMap<(usize, usize), Scalar>;
type PositionMatrixRefTriple<'a, Scalar> = (&'a PositionMatrix<Scalar>, Realness, Canonicality);
type PositionMatrixMutTriple<'a, Scalar> = (&'a mut PositionMatrix<Scalar>, Realness, Canonicality);

type PositionMatrixRowColDataFormat<Scalar> = (Vec<usize>, Vec<usize>, Vec<Scalar>);

#[pyclass(frozen, module = "ncpoleon.relaxations", eq, eq_int, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Realness {
    Real,
    Complex,
}

#[pyclass(frozen, module = "ncpoleon.relaxations", eq, eq_int, skip_from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Canonicality {
    Canonical,
    Adjoint,
}

fn position_matrix_to_row_col_data_format<Scalar: PolynomialDtype>(
    position_matrix: &PositionMatrix<Scalar>,
) -> PositionMatrixRowColDataFormat<Scalar> {
    // The capacity is the number of stored entries, not the side of the matrix: a position matrix
    // holds up to `size * size` of them
    let mut rows = Vec::with_capacity(position_matrix.len());
    let mut cols = Vec::with_capacity(position_matrix.len());
    let mut data = Vec::with_capacity(position_matrix.len());

    for (index, &value) in position_matrix.iter() {
        rows.push(index.0);
        cols.push(index.1);
        data.push(value);
    }

    (rows, cols, data)
}

#[derive(Clone)]
pub(super) struct RustMomentMatrix<Scalar: PolynomialDtype, MonomialType: AdjointTrait + Ord> {
    pub(super) associated_id: u8,
    /// The realness is needed to interpret the adjoints. For instance, if the problem is
    /// complex-valued, but X_1 and X_2 are Hermitian, then the entries for X_1X_2 and X_2X_1 are
    /// conjugate of each other, so we shouldn't create a new variable for X_2X_1, but instead reuse
    /// the one used for X_1X_2 and conjugate it. For real-valued problems, the conjugate is equal
    /// to the base value, so there's no need to store it. In both cases, we only need to know the
    /// positions of the canonical monomial to know that of its adjoint.
    data: BTreeMap<MonomialType, (PositionMatrix<Scalar>, Realness)>,
    /// Maps the canonical form of each stored key's adjoint back to that key. Maintained by
    /// [`RustMomentMatrix::insert`], it lets [`RustMomentMatrix::get_mut`] resolve a monomial stored
    /// under the opposite (adjoint) orientation without recomputing `adjoint().rewrite()` on every
    /// lookup.
    adjoint_index: BTreeMap<MonomialType, MonomialType>,
    pub(super) size: usize,
}

impl<Scalar, MonomialType> RustMomentMatrix<Scalar, MonomialType>
where
    Scalar: PolynomialDtype,
    MonomialType: AdjointTrait + Ord + RewritingTrait<MonomialType> + Display + Clone,
{
    pub(super) fn new(associated_id: u8, size: usize) -> Self {
        Self { associated_id, data: BTreeMap::new(), adjoint_index: BTreeMap::new(), size }
    }

    /// Insert a fresh entry for `monomial`, registering its adjoint's canonical form in
    /// `adjoint_index` so that later [`get_mut`](Self::get_mut) lookups for the opposite orientation
    /// resolve without recomputing `adjoint().rewrite()`. This is the only place the adjoint of a
    /// stored key is rewritten, and it also rejects `monomial` if rewriting turns out not to be
    /// adjoint-stable on it (see below)
    pub(super) fn insert(
        &mut self,
        monomial: MonomialType,
        position_matrix: PositionMatrix<Scalar>,
        realness: Realness,
        strategy: RewritingStrategy,
        substitutions: &BTreeMap<MonomialType, MonomialType>,
    ) -> Result<(), String> {
        let adjoint = monomial.adjoint().rewrite(strategy, substitutions)?;

        // `get_mut` resolves the opposite orientation through `adjoint_index`, which agrees with
        // `get`/`get_canonical` (they rewrite the adjoint of the *query*) only if `rewrite ∘ adjoint`
        // is an involution on canonical monomials. That holds when the rewriting system is confluent
        // *and* its rule set is closed under adjoint; the substitutions come straight from the user,
        // so neither is currently guaranteed.
        // TODO: once rewriting goes through a completed (confluent, adjoint-closed) rule set, this
        //  invariant holds by construction and the check (along with the extra rewrite it costs per
        //  stored monomial) can be dropped.
        let roundtrip = adjoint.adjoint().rewrite(strategy, substitutions)?;
        if roundtrip != monomial {
            return Err(format!(
                "Rewriting is not adjoint-stable: {} has adjoint {}, whose adjoint rewrites back to \
                 {} instead. This usually means the substitution rules are not closed under adjoint \
                 (a rule l -> r was given without its counterpart l* -> r*).",
                monomial, adjoint, roundtrip
            ));
        }

        if self.data.contains_key(&monomial) {
            return Err(format!("Trying to insert the already present {} in a moment matrix.", monomial));
        }

        if self.adjoint_index.contains_key(&monomial) {
            return Err(format!(
                "Trying to insert {} in a moment matrix when its adjoint has already been inserted.",
                monomial
            ));
        }

        self.adjoint_index.insert(adjoint, monomial.clone());
        self.data.insert(monomial, (position_matrix, realness));
        Ok(())
    }

    /// Look up the entry a monomial belongs to. A query resolves iff it is a stored key (direct
    /// orientation) or the adjoint-canonical of a stored key (via `adjoint_index`); no rewriting is
    /// performed here, so the query must already be in canonical form. Entries must be added through
    /// [`insert`](Self::insert) for `adjoint_index` to stay consistent.
    pub(super) fn get(&self, monomial: &MonomialType) -> Result<Option<PositionMatrixRefTriple<'_, Scalar>>, String> {
        if let Some((position_matrix, realness)) = self.data.get(monomial) {
            return Ok(Some((position_matrix, *realness, Canonicality::Canonical)));
        }
        match self.adjoint_index.get(monomial) {
            Some(canonical) => {
                let (position_matrix, realness) = self.data.get(canonical).ok_or_else(|| {
                    format!(
                        "The adjoint {} of the canonical monomial {} is present within the adjoint \
                        index of this moment matrix, but no entry is associated to this canonical \
                        monomial. This is likely a mistake on our side, so feel free to open an \
                        issue about this!",
                        monomial, canonical
                    )
                })?;
                Ok(Some((position_matrix, *realness, Canonicality::Adjoint)))
            }
            None => Ok(None),
        }
    }

    /// Look up the entry a monomial belongs to. A query resolves iff it is a stored key (direct
    /// orientation) or the adjoint-canonical of a stored key (via `adjoint_index`); no rewriting is
    /// performed here, so the query must already be in canonical form. Entries must be added through
    /// [`insert`](Self::insert) for `adjoint_index` to stay consistent.
    pub(super) fn get_mut(
        &mut self,
        monomial: &MonomialType,
    ) -> Result<Option<PositionMatrixMutTriple<'_, Scalar>>, String> {
        if self.data.contains_key(monomial) {
            let (position_matrix, realness) = self.data.get_mut(monomial).unwrap();
            return Ok(Some((position_matrix, *realness, Canonicality::Canonical)));
        }
        match self.adjoint_index.get(monomial) {
            Some(canonical) => {
                let (position_matrix, realness) = self.data.get_mut(canonical).ok_or_else(|| {
                    format!(
                        "The adjoint {} of the canonical monomial {} is present within the adjoint \
                        index of this moment matrix, but no entry is associated to this canonical \
                        monomial. This is likely a mistake on our side, so feel free to open an \
                        issue about this!",
                        monomial, canonical
                    )
                })?;
                Ok(Some((position_matrix, *realness, Canonicality::Adjoint)))
            }
            None => Ok(None),
        }
    }

    /// get_canonical is used to verify that a monomial or its adjoint are stored. If neither are stored, it raises
    /// an Error.
    pub(super) fn get_canonical(
        &self,
        monomial: &MonomialType,
    ) -> Result<(MonomialType, Canonicality, Realness), String> {
        let (_, realness, canonicality) = self
            .get(monomial)?
            .ok_or_else(|| format!("Couldn't find monomial {} or its adjoint in the moment matrix.", monomial))?;
        match canonicality {
            Canonicality::Canonical => Ok((monomial.clone(), canonicality, realness)),
            // We can afford to use unwrap here `get` already performs this check
            Canonicality::Adjoint => Ok((self.adjoint_index.get(monomial).unwrap().clone(), canonicality, realness)),
        }
    }
}

type RustRealValuedMomentMatrix<MonomialType> = RustMomentMatrix<f64, MonomialType>;
type RustComplexValuedMomentMatrix<MonomialType> = RustMomentMatrix<Complex<f64>, MonomialType>;

#[pyclass(
    frozen,
    module = "ncpoleon.relaxations",
    name = "RealValuedCommutativeMomentMatrix",
    mapping,
    skip_from_py_object
)]
#[derive(Clone)]
pub(super) struct PythonRealValuedCommutativeMomentMatrix(
    pub(super) RustRealValuedMomentMatrix<RustCommutativeMonomial>,
);

#[pyclass(
    frozen,
    module = "ncpoleon.relaxations",
    name = "ComplexValuedCommutativeMomentMatrix",
    mapping,
    skip_from_py_object
)]
#[derive(Clone)]
pub(super) struct PythonComplexValuedCommutativeMomentMatrix(
    pub(super) RustComplexValuedMomentMatrix<RustCommutativeMonomial>,
);

#[pyclass(
    frozen,
    module = "ncpoleon.relaxations",
    name = "RealValuedNonCommutativeMomentMatrix",
    mapping,
    skip_from_py_object
)]
#[derive(Clone)]
pub(super) struct PythonRealValuedNonCommutativeMomentMatrix(
    pub(super) RustRealValuedMomentMatrix<RustNonCommutativeMonomial>,
);

#[pyclass(
    frozen,
    module = "ncpoleon.relaxations",
    name = "ComplexValuedNonCommutativeMomentMatrix",
    mapping,
    skip_from_py_object
)]
#[derive(Clone)]
pub(super) struct PythonComplexValuedNonCommutativeMomentMatrix(
    pub(super) RustComplexValuedMomentMatrix<RustNonCommutativeMonomial>,
);

macro_rules! impl_moment_matrix_pymethods {
    ($py_moment_matrix:ident, $py_monomial:ident, $scalar_type:ty) => {
        #[pymethods]
        impl $py_moment_matrix {
            #[getter]
            fn size(&self) -> usize {
                self.0.size
            }

            pub(super) fn as_row_col_data_format(
                &self,
            ) -> BTreeMap<$py_monomial, (PositionMatrixRowColDataFormat<$scalar_type>, Realness)> {
                BTreeMap::from_iter(self.0.data.iter().map(|(monomial, (position_matrix, realness))| {
                    (
                        $py_monomial(monomial.clone()),
                        (position_matrix_to_row_col_data_format(position_matrix), *realness),
                    )
                }))
            }

            fn __contains__<'py>(&self, item: &Bound<'py, PyAny>) -> PyResult<bool> {
                let rust_monomial = $py_monomial::try_from_reference_bound(item, Some(self.0.associated_id))?.0;
                Ok(self.0.get_canonical(&rust_monomial).is_ok())
            }

            fn get_canonical<'py>(
                &self,
                monomial: &Bound<'py, PyAny>,
            ) -> PyResult<($py_monomial, Canonicality, Realness)> {
                let python_monomial = $py_monomial::try_from_reference_bound(monomial, Some(self.0.associated_id))?;
                let (rust_monomial, canonicality, realness) =
                    self.0.get_canonical(&python_monomial.0).map_err(PyKeyError::new_err)?;
                Ok(($py_monomial(rust_monomial), canonicality, realness))
            }
        }
    };
}

impl_moment_matrix_pymethods!(PythonRealValuedCommutativeMomentMatrix, PythonCommutativeMonomial, f64);

impl_moment_matrix_pymethods!(PythonComplexValuedCommutativeMomentMatrix, PythonCommutativeMonomial, Complex<f64>);

impl_moment_matrix_pymethods!(PythonRealValuedNonCommutativeMomentMatrix, PythonNonCommutativeMonomial, f64);

impl_moment_matrix_pymethods!(
    PythonComplexValuedNonCommutativeMomentMatrix,
    PythonNonCommutativeMonomial,
    Complex<f64>
);
