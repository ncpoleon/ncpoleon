use std::collections::BTreeMap;
use std::fmt::Display;

use num_complex::Complex;
use pyo3::exceptions::{PyKeyError, PyValueError};
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
type PositionMatrixPair<Scalar> = (PositionMatrix<Scalar>, Option<PositionMatrix<Scalar>>);
type PositionMatrixRefPair<'a, Scalar> = (&'a PositionMatrix<Scalar>, Option<&'a PositionMatrix<Scalar>>);
type PositionMatrixMutPair<'a, Scalar> = (&'a mut PositionMatrix<Scalar>, Option<&'a mut PositionMatrix<Scalar>>);

type PositionMatrixRowColDataFormat<Scalar> = (Vec<usize>, Vec<usize>, Vec<Scalar>);

fn position_matrix_to_row_col_data_format<Scalar: PolynomialDtype>(
    position_matrix: &PositionMatrix<Scalar>,
    size: usize,
) -> PositionMatrixRowColDataFormat<Scalar> {
    let mut rows = Vec::with_capacity(size);
    let mut cols = Vec::with_capacity(size);
    let mut data = Vec::with_capacity(size);

    for (index, &value) in position_matrix.iter() {
        rows.push(index.0);
        cols.push(index.1);
        data.push(value);
    }

    (rows, cols, data)
}

// The second position matrix is that of the conjugate if the problem is complex-valued. For instance, if the
// problem is complex-valued, but X_1 and X_2 are Hermitian, then the entries for X_1X_2 and X_2X_1 are conjugate
// of each other, so we shouldn't create a new variable for X_2X_1, but instead reuse the one used for X_1X_2 and
// conjugate it. For real-valued problems, the conjugate is equal to the base value, so there's no need to store it.
// FIXME: we don't really have to store the adjoint positions in case of a complex-valued problem, since this will
//  always correspond to the symmetry of the canonical w.r.t. the diagonal. A simple flag to indicate whether the
//  variable is complex could suffice, modulo some changes in the code.
#[derive(Clone)]
pub(super) struct RustMomentMatrix<Scalar: PolynomialDtype, MonomialType: AdjointTrait + Ord> {
    pub(super) data: BTreeMap<MonomialType, PositionMatrixPair<Scalar>>,
    pub(super) size: usize,
}

impl<Scalar, MonomialType> RustMomentMatrix<Scalar, MonomialType>
where
    Scalar: PolynomialDtype,
    MonomialType: AdjointTrait + Ord + RewritingTrait<MonomialType> + Display + Clone,
{
    pub(super) fn get(
        &self,
        monomial: &MonomialType,
        strategy: RewritingStrategy,
        substitutions: &BTreeMap<MonomialType, MonomialType>,
    ) -> Result<Option<PositionMatrixRefPair<'_, Scalar>>, String> {
        if self.data.contains_key(monomial) {
            let (position_matrix, position_matrix_conj) = self.data.get(monomial).unwrap();
            return Ok(Some((position_matrix, position_matrix_conj.as_ref())));
        }
        let adjoint = monomial.adjoint().rewrite(strategy, substitutions)?;
        if self.data.contains_key(&adjoint) {
            let (position_matrix_conj, position_matrix) = self.data.get(&adjoint).unwrap();
            return Ok(Some(match position_matrix {
                Some(position_matrix) => (position_matrix, Some(position_matrix_conj)),
                None => (position_matrix_conj, None),
            }));
        }
        Ok(None)
    }

    pub(super) fn get_mut(
        &mut self,
        monomial: &MonomialType,
        strategy: RewritingStrategy,
        substitutions: &BTreeMap<MonomialType, MonomialType>,
    ) -> Result<Option<PositionMatrixMutPair<'_, Scalar>>, String> {
        if self.data.contains_key(monomial) {
            let (position_matrix, position_matrix_conj) = self.data.get_mut(monomial).unwrap();
            return Ok(Some((position_matrix, position_matrix_conj.as_mut())));
        }
        let adjoint = monomial.adjoint().rewrite(strategy, substitutions)?;
        if self.data.contains_key(&adjoint) {
            let (position_matrix_conj, position_matrix) = self.data.get_mut(&adjoint).unwrap();
            return Ok(Some(match position_matrix {
                Some(position_matrix) => (position_matrix, Some(position_matrix_conj)),
                None => (position_matrix_conj, None),
            }));
        }
        Ok(None)
    }

    /// get_canonical is used to verify that a monomial or its adjoint are stored. If neither are stored, it raises
    /// an Error. Otherwise, the first return parameter is the canonical form that is stored, the second is whether
    /// we had to take the adjoint, and the third is whether the corresponding moment is real-valued (i.e. the entry
    /// is stored as a single position matrix without a separate conjugate matrix).
    pub(super) fn get_canonical(
        &self,
        monomial: &MonomialType,
        strategy: RewritingStrategy,
        substitutions: &BTreeMap<MonomialType, MonomialType>,
    ) -> Result<(MonomialType, bool, bool), String> {
        if let Some((_, position_matrix_conj)) = self.data.get(monomial) {
            return Ok((monomial.clone(), false, position_matrix_conj.is_none()));
        }
        let adjoint = monomial.adjoint().rewrite(strategy, substitutions)?;
        if let Some((_, position_matrix_conj)) = self.data.get(&adjoint) {
            return Ok((adjoint, true, position_matrix_conj.is_none()));
        }
        Err(format!("Couldn't find monomial {} or its adjoint in the moment matrix.", monomial))
    }
}

type RustRealValuedMomentMatrix<MonomialType> = RustMomentMatrix<f64, MonomialType>;
type RustComplexValuedMomentMatrix<MonomialType> = RustMomentMatrix<Complex<f64>, MonomialType>;

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "RealValuedCommutativeMomentMatrix", mapping)]
#[derive(Clone)]
pub(super) struct PythonRealValuedCommutativeMomentMatrix(
    pub(super) RustRealValuedMomentMatrix<RustCommutativeMonomial>,
);

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "ComplexValuedCommutativeMomentMatrix", mapping)]
#[derive(Clone)]
pub(super) struct PythonComplexValuedCommutativeMomentMatrix(
    pub(super) RustComplexValuedMomentMatrix<RustCommutativeMonomial>,
);

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "RealValuedNonCommutativeMomentMatrix", mapping)]
#[derive(Clone)]
pub(super) struct PythonRealValuedNonCommutativeMomentMatrix(
    pub(super) RustRealValuedMomentMatrix<RustNonCommutativeMonomial>,
);

#[pyclass(frozen, module = "ncpoleon.relaxations", name = "ComplexValuedNonCommutativeMomentMatrix", mapping)]
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
            ) -> BTreeMap<
                $py_monomial,
                (PositionMatrixRowColDataFormat<$scalar_type>, Option<PositionMatrixRowColDataFormat<$scalar_type>>),
            > {
                BTreeMap::from_iter(self.0.data.iter().map(|(monomial, (position_matrix, position_matrix_conj))| {
                    (
                        $py_monomial(monomial.clone()),
                        (
                            position_matrix_to_row_col_data_format(position_matrix, self.0.size),
                            position_matrix_conj
                                .as_ref()
                                .map(|pos_matrix| position_matrix_to_row_col_data_format(pos_matrix, self.0.size)),
                        ),
                    )
                }))
            }

            fn __contains__<'py>(&self, item: &Bound<'py, PyAny>) -> bool {
                let rust_monomial: Result<$py_monomial, PyErr> = item.try_into();
                rust_monomial.is_ok()
            }

            fn __getitem__<'py>(
                &self,
                key: &Bound<'py, PyAny>,
            ) -> PyResult<PositionMatrixRowColDataFormat<$scalar_type>> {
                let python_monomial: $py_monomial = key.try_into()?;
                let res = self
                    .0
                    .get(&python_monomial.0, RewritingStrategy::None, &BTreeMap::new())
                    .map_err(PyValueError::new_err)?;
                match res {
                    Some((pos_matrix, _position_matrix_conj)) => {
                        Ok(position_matrix_to_row_col_data_format(pos_matrix, self.0.size))
                    }
                    None => Err(PyKeyError::new_err(format!(
                        "Couldn't find monomial {} in the moment matrix.",
                        python_monomial.0
                    ))),
                }
            }

            fn get_canonical(&self, monomial: &$py_monomial) -> PyResult<($py_monomial, bool, bool)> {
                let (rust_monomial, is_adjoint, is_real_valued) = self
                    .0
                    .get_canonical(&monomial.0, RewritingStrategy::None, &BTreeMap::new())
                    .map_err(PyValueError::new_err)?;
                Ok(($py_monomial(rust_monomial), is_adjoint, is_real_valued))
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
