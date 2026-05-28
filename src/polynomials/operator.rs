use std::cmp::Ordering;
use std::ops::{Add, Sub};

use crate::polynomials::monomial::{HasAMomentMatrixId, Monomial};
use crate::polynomials::polynomial::{Polynomial, PolynomialDtype};

#[derive(Clone, Debug, Copy)]
pub(crate) struct Operator<Id> {
    pub(crate) id: Id,
}

impl<Id: HasAMomentMatrixId> HasAMomentMatrixId for Operator<Id> {
    fn moment_matrix_id(&self) -> u8 {
        self.id.moment_matrix_id()
    }
}

impl<I> PartialEq<Self> for Operator<I>
where
    I: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<I> Eq for Operator<I> where I: PartialEq {}

impl<I> Ord for Operator<I>
where
    I: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl<I> PartialOrd for Operator<I>
where
    I: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, M> Add<&Monomial<M>> for &Operator<I>
where
    for<'a> Monomial<M>: From<&'a Operator<I>>,
    M: Ord + Clone,
{
    type Output = Polynomial<Monomial<M>, f64>;

    fn add(self, rhs: &Monomial<M>) -> Self::Output {
        Monomial::from(self) + rhs
    }
}

impl<I, T, M> Add<&Polynomial<Monomial<M>, T>> for &Operator<I>
where
    for<'a> Monomial<M>: From<&'a Operator<I>>,
    T: PolynomialDtype,
    M: Ord + Clone,
{
    type Output = Polynomial<Monomial<M>, T>;

    fn add(self, rhs: &Polynomial<Monomial<M>, T>) -> Self::Output {
        Monomial::from(self) + rhs
    }
}

impl<I, M> Sub<&Monomial<M>> for &Operator<I>
where
    for<'a> Monomial<M>: From<&'a Operator<I>>,
    M: Ord + Clone,
{
    type Output = Polynomial<Monomial<M>, f64>;

    fn sub(self, rhs: &Monomial<M>) -> Self::Output {
        Monomial::from(self) - rhs
    }
}

impl<I, T, M> Sub<&Polynomial<Monomial<M>, T>> for &Operator<I>
where
    for<'a> Monomial<M>: From<&'a Operator<I>>,
    T: PolynomialDtype,
    M: Ord + Clone,
{
    type Output = Polynomial<Monomial<M>, T>;

    fn sub(self, rhs: &Polynomial<Monomial<M>, T>) -> Self::Output {
        Monomial::from(self) - rhs
    }
}
