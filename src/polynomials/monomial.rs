use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::{Add, Mul, Neg, Sub};

use num_traits::{Pow, Zero};
use pyo3::prelude::*;

use crate::polynomials::operator::Operator;
use crate::polynomials::polynomial::{Polynomial, PolynomialDtype};
use crate::polynomials::utils::add::manage_entry;
use crate::polynomials::utils::pow::fallible_exponentiation_by_squaring;

pub(crate) trait RewritingTrait<MonomialKind> {
    fn rewrite(
        &self,
        strategy: RewritingStrategy,
        substitutions: &BTreeMap<MonomialKind, MonomialKind>,
    ) -> Result<Self, String>
    where
        Self: Sized;
}

pub(crate) trait AdjointTrait {
    fn adjoint(&self) -> Self;
}

pub(crate) trait HasAMomentMatrixId {
    fn moment_matrix_id(&self) -> u8;
}

pub(crate) trait OneWithMomentMatrixId: HasAMomentMatrixId {
    fn one(moment_matrix_id: u8) -> Self;
    fn is_one(&self) -> bool;
    fn identity_symbol() -> Option<&'static str> {
        None
    }
}

impl<Data: HasAMomentMatrixId> HasAMomentMatrixId for Monomial<Data> {
    fn moment_matrix_id(&self) -> u8 {
        self.data.moment_matrix_id()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Monomial<Data> {
    pub(crate) data: Data,
}

/// Strategy used when rewriting monomials via substitutions.
///
/// The strategy controls how the substitution rules are applied when
/// simplifying or normalizing monomials.
#[pyclass(frozen, module = "ncpoleon.polynomials")]
#[derive(Clone, Copy)]
pub(crate) enum RewritingStrategy {
    /// Do not perform any substitution
    None,
    /// Apply substitution rules greedily: at each step, pick the matching
    /// rule that results in the largest degree change and apply it. Repeat
    /// until no more rules match.
    Greedy,
}

impl<D> PartialEq<Self> for Monomial<D>
where
    D: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<D> Eq for Monomial<D> where D: PartialEq {}

impl<D> Ord for Monomial<D>
where
    D: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}

impl<D> PartialOrd for Monomial<D>
where
    D: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Data> Pow<u8> for &Monomial<Data>
where
    Monomial<Data>: OneWithMomentMatrixId,
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    Data: Ord + Clone,
{
    type Output = Result<Monomial<Data>, String>;

    fn pow(self, rhs: u8) -> Self::Output {
        fallible_exponentiation_by_squaring(self, Monomial::one(self.moment_matrix_id()), rhs as usize)
    }
}

impl<D, T> Add<T> for &Monomial<D>
where
    Monomial<D>: Clone + OneWithMomentMatrixId,
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Result<Polynomial<Monomial<D>, T>, String>;

    fn add(self, rhs: T) -> Self::Output {
        Polynomial::from(self) + rhs
    }
}

impl<D, T> Add<T> for Monomial<D>
where
    Monomial<D>: Clone + OneWithMomentMatrixId,
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Result<Polynomial<Monomial<D>, T>, String>;

    fn add(self, rhs: T) -> Self::Output {
        Polynomial::from(self) + rhs
    }
}

impl<D, T> Sub<T> for &Monomial<D>
where
    Monomial<D>: Clone + OneWithMomentMatrixId,
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Result<Polynomial<Monomial<D>, T>, String>;

    fn sub(self, rhs: T) -> Self::Output {
        Polynomial::from(self) - rhs
    }
}

impl<D, T> Sub<T> for Monomial<D>
where
    Monomial<D>: Clone + OneWithMomentMatrixId,
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Result<Polynomial<Monomial<D>, T>, String>;

    fn sub(self, rhs: T) -> Self::Output {
        Polynomial::from(self) - rhs
    }
}

impl<I, M> Add<&Operator<I>> for &Monomial<M>
where
    for<'a> Monomial<M>: From<&'a Operator<I>>,
    M: Ord + Clone,
{
    type Output = Polynomial<Monomial<M>, f64>;

    fn add(self, rhs: &Operator<I>) -> Self::Output {
        self + Monomial::from(rhs)
    }
}

impl<I, M> Sub<&Operator<I>> for &Monomial<M>
where
    for<'a> Monomial<M>: From<&'a Operator<I>>,
    M: Ord + Clone,
{
    type Output = Polynomial<Monomial<M>, f64>;

    fn sub(self, rhs: &Operator<I>) -> Self::Output {
        self - Monomial::from(rhs)
    }
}

impl<D> Add<&Monomial<D>> for &Monomial<D>
where
    Monomial<D>: Clone,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn add(self, rhs: &Monomial<D>) -> Self::Output {
        Polynomial::from(self) + Polynomial::from(rhs)
    }
}

impl<D> Add<&Monomial<D>> for Monomial<D>
where
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn add(self, rhs: &Monomial<D>) -> Self::Output {
        Polynomial::from(self) + Polynomial::from(rhs)
    }
}

impl<D> Add<Monomial<D>> for &Monomial<D>
where
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn add(self, rhs: Monomial<D>) -> Self::Output {
        Polynomial::from(self) + Polynomial::from(rhs)
    }
}

impl<D> Add<Monomial<D>> for Monomial<D>
where
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn add(self, rhs: Monomial<D>) -> Self::Output {
        Polynomial::from(self) + Polynomial::from(rhs)
    }
}

impl<D, T> Add<&Polynomial<Monomial<D>, T>> for &Monomial<D>
where
    Monomial<D>: Clone,
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn add(self, rhs: &Polynomial<Monomial<D>, T>) -> Self::Output {
        Polynomial::<Monomial<D>, T>::from(self) + rhs
    }
}

impl<D, T> Add<&Polynomial<Monomial<D>, T>> for Monomial<D>
where
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn add(self, rhs: &Polynomial<Monomial<D>, T>) -> Self::Output {
        Polynomial::<Monomial<D>, T>::from(self) + rhs
    }
}

impl<D, T> Add<Polynomial<Monomial<D>, T>> for Monomial<D>
where
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn add(self, rhs: Polynomial<Monomial<D>, T>) -> Self::Output {
        Polynomial::<Monomial<D>, T>::from(self) + rhs
    }
}

impl<D> Sub<&Monomial<D>> for &Monomial<D>
where
    Monomial<D>: Clone,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn sub(self, rhs: &Monomial<D>) -> Self::Output {
        Polynomial::from(self) - Polynomial::from(rhs)
    }
}

impl<D> Sub<Monomial<D>> for &Monomial<D>
where
    Monomial<D>: Clone,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn sub(self, rhs: Monomial<D>) -> Self::Output {
        Polynomial::from(self) - Polynomial::from(rhs)
    }
}

impl<D> Sub<&Monomial<D>> for Monomial<D>
where
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn sub(self, rhs: &Monomial<D>) -> Self::Output {
        Polynomial::from(self) - Polynomial::from(rhs)
    }
}

impl<D> Sub<Monomial<D>> for Monomial<D>
where
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, f64>;

    fn sub(self, rhs: Monomial<D>) -> Self::Output {
        Polynomial::from(self) - Polynomial::from(rhs)
    }
}

impl<D, T> Sub<&Polynomial<Monomial<D>, T>> for &Monomial<D>
where
    Monomial<D>: Clone,
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn sub(self, rhs: &Polynomial<Monomial<D>, T>) -> Self::Output {
        Polynomial::<Monomial<D>, T>::from(self) - rhs
    }
}

impl<D, T> Sub<&Polynomial<Monomial<D>, T>> for Monomial<D>
where
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn sub(self, rhs: &Polynomial<Monomial<D>, T>) -> Self::Output {
        Polynomial::<Monomial<D>, T>::from(self) - rhs
    }
}

impl<D, T> Sub<Polynomial<Monomial<D>, T>> for Monomial<D>
where
    T: PolynomialDtype,
    D: Ord + Clone,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn sub(self, rhs: Polynomial<Monomial<D>, T>) -> Self::Output {
        Polynomial::<Monomial<D>, T>::from(self) - rhs
    }
}

impl<D, T> Mul<T> for &Monomial<D>
where
    D: Ord + Clone,
    T: PolynomialDtype,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn mul(self, rhs: T) -> Self::Output {
        if rhs.is_zero() { Polynomial::zero() } else { Polynomial { data: BTreeMap::from([(self.clone(), rhs)]) } }
    }
}

impl<D, T> Mul<T> for Monomial<D>
where
    D: Ord + Clone,
    T: PolynomialDtype,
{
    type Output = Polynomial<Monomial<D>, T>;

    fn mul(self, rhs: T) -> Self::Output {
        if rhs.is_zero() { Polynomial::zero() } else { Polynomial { data: BTreeMap::from([(self, rhs)]) } }
    }
}

impl<Data> Neg for &Monomial<Data>
where
    Data: Clone + Ord,
{
    type Output = Polynomial<Monomial<Data>, f64>;

    fn neg(self) -> Self::Output {
        -Polynomial::from(self)
    }
}

impl<Data> Neg for Monomial<Data>
where
    Data: Clone + Ord,
{
    type Output = Polynomial<Monomial<Data>, f64>;

    fn neg(self) -> Self::Output {
        -Polynomial::from(self)
    }
}

impl<Data, Scalar> Mul<&Polynomial<Monomial<Data>, Scalar>> for Monomial<Data>
where
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Self::Output {
        let mut res = BTreeMap::new();

        for (mon, &coeff) in rhs.data.iter() {
            manage_entry(&mut res, (&self * mon)?, coeff);
        }

        Ok(Polynomial { data: res })
    }
}
