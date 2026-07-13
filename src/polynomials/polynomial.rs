use std::collections::BTreeMap;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

use log::{trace, warn};
use num_complex::Complex;
use num_traits::{One, Pow, Zero};

use crate::polynomials::monomial::{
    AdjointTrait, HasAMomentMatrixId, Monomial, OneWithMomentMatrixId, RewritingStrategy, RewritingTrait,
};
use crate::polynomials::operator::Operator;
use crate::polynomials::utils::add::manage_entry;
use crate::polynomials::utils::merge_btreemaps::merge_btreemaps;
use crate::polynomials::utils::mul::poly_mul;
use crate::polynomials::utils::pow::fallible_exponentiation_by_squaring;

pub(crate) trait Conjugate {
    fn conjugate(&self) -> Self;
}

pub(crate) trait TryIntoReal {
    fn try_into_real(&self) -> Result<f64, String>;
}

pub(crate) trait PolynomialDtype:
    Clone
    + Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
    + Zero
    + One
    + PartialEq
    + Display
    + Conjugate
    + TryIntoReal
{
}

impl Conjugate for f64 {
    fn conjugate(&self) -> f64 {
        *self
    }
}

impl Conjugate for Complex<f64> {
    fn conjugate(&self) -> Complex<f64> {
        self.conj()
    }
}

impl TryIntoReal for f64 {
    fn try_into_real(&self) -> Result<f64, String> {
        Ok(*self)
    }
}

impl TryIntoReal for Complex<f64> {
    fn try_into_real(&self) -> Result<f64, String> {
        if self.im == 0f64 {
            Ok(self.re)
        } else {
            Err(format!(
                "Can't convert complex number {}+i{} with non-zero imaginary part to a real number.",
                self.re, self.im
            ))
        }
    }
}

impl PolynomialDtype for f64 {}
impl PolynomialDtype for Complex<f64> {}

pub(crate) trait PolynomialTrait: AdjointTrait {
    fn chop(&self, delta: f64) -> Self;
    fn degree(&self) -> u8;
    fn is_real(&self) -> bool;
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Polynomial<MonomialType, Scalar: PolynomialDtype> {
    pub(crate) data: BTreeMap<MonomialType, Scalar>,
}

impl<MonomialType, Scalar: PolynomialDtype> Polynomial<MonomialType, Scalar>
where
    MonomialType: HasAMomentMatrixId + Ord + Clone,
{
    pub(crate) fn get_unique_moment_matrix_id(&self) -> Option<u8> {
        let mut data_iter = self.data.iter();

        if let Some((first_mon, _)) = data_iter.next() {
            let first_index = first_mon.moment_matrix_id();
            if data_iter.all(|(mon, _)| mon.moment_matrix_id() == first_index) { Some(first_index) } else { None }
        } else {
            None
        }
    }

    pub(crate) fn by_moment_matrix_id(&self) -> BTreeMap<u8, Self> {
        let mut res_as_btree_maps = BTreeMap::new();

        for (monomial, &coefficient) in self.data.iter() {
            res_as_btree_maps
                .entry(monomial.moment_matrix_id())
                .and_modify(|internal_map: &mut BTreeMap<MonomialType, Scalar>| {
                    internal_map.insert(monomial.clone(), coefficient);
                })
                .or_insert(BTreeMap::from([(monomial.clone(), coefficient)]));
        }

        res_as_btree_maps.into_iter().map(|(moment_matrix_id, data)| (moment_matrix_id, Polynomial { data })).collect()
    }
}

impl<Data, Scalar: PolynomialDtype> RewritingTrait<Monomial<Data>> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Monomial<Data>: RewritingTrait<Monomial<Data>>,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    /// Rewrite a polynomial using the given rewriting strategy and a set of substitution rules on
    /// monomials
    fn rewrite(
        &self,
        strategy: RewritingStrategy,
        substitutions: &BTreeMap<Monomial<Data>, Monomial<Data>>,
    ) -> Result<Self, String> {
        if self.data.is_empty() {
            return Ok(Self::zero());
        }
        trace!("Rewriting polynomial {}.", self);
        let result = self
            .data
            .iter()
            .map(|(monomial, &coeff)| {
                Ok(Self { data: BTreeMap::from([(monomial.rewrite(strategy, substitutions)?, coeff)]) })
            })
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            // Reducing like this allows to simplify the polynomial by removing opposite terms
            .reduce(|poly1, poly2| poly1 + poly2)
            .unwrap();
        trace!("Rewrote polynomial into {}.", result);
        Ok(result)
    }
}

impl<Data> Display for Polynomial<Monomial<Data>, f64>
where
    Monomial<Data>: Display + OneWithMomentMatrixId + Ord + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (monom, coeff) in &self.data {
            let (prefix, abs_coeff) = if *coeff < 0.0 {
                (if first { "-" } else { " - " }, -coeff)
            } else {
                (if first { "" } else { " + " }, *coeff)
            };
            write!(f, "{}", prefix)?;
            if monom.is_one() {
                match Monomial::<Data>::identity_symbol() {
                    None => write!(f, "{}", abs_coeff)?,
                    Some(sym) if abs_coeff == 1.0 => write!(f, "{}", sym)?,
                    Some(sym) => write!(f, "{} * {}", abs_coeff, sym)?,
                }
            } else if abs_coeff == 1.0 {
                write!(f, "{}", monom)?;
            } else {
                write!(f, "{} * {}", abs_coeff, monom)?;
            }
            first = false;
        }
        Ok(())
    }
}

impl<Data> Display for Polynomial<Monomial<Data>, Complex<f64>>
where
    Monomial<Data>: Display + OneWithMomentMatrixId + Ord + PartialEq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.data.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (monom, coeff) in &self.data {
            let (is_negative, abs_coeff) = if coeff.im == 0.0 && coeff.re < 0.0 {
                (true, Complex::new(-coeff.re, 0.0))
            } else if coeff.re == 0.0 && coeff.im < 0.0 {
                (true, Complex::new(0.0, -coeff.im))
            } else {
                (false, *coeff)
            };
            let prefix = if is_negative {
                if first { "- " } else { " - " }
            } else if first {
                ""
            } else {
                " + "
            };
            write!(f, "{}", prefix)?;
            let format_scalar = |f: &mut fmt::Formatter<'_>| -> fmt::Result {
                if abs_coeff.im == 0.0 {
                    write!(f, "{}", abs_coeff.re)
                } else if abs_coeff.re == 0.0 {
                    write!(f, "{}i", abs_coeff.im)
                } else {
                    write!(f, "{}", abs_coeff)
                }
            };
            if monom.is_one() {
                match Monomial::<Data>::identity_symbol() {
                    None => format_scalar(f)?,
                    Some(sym) if abs_coeff == Complex::ONE => write!(f, "{}", sym)?,
                    Some(sym) if abs_coeff.im == 0.0 => write!(f, "{} * {}", abs_coeff.re, sym)?,
                    Some(sym) if abs_coeff.re == 0.0 => write!(f, "{}i * {}", abs_coeff.im, sym)?,
                    Some(sym) => write!(f, "({}) * {}", abs_coeff, sym)?,
                }
            } else if abs_coeff == Complex::ONE {
                write!(f, "{}", monom)?;
            } else if abs_coeff.im == 0.0 {
                write!(f, "{} * {}", abs_coeff.re, monom)?;
            } else if abs_coeff.re == 0.0 {
                write!(f, "{}i * {}", abs_coeff.im, monom)?;
            } else {
                write!(f, "({}) * {}", abs_coeff, monom)?;
            }
            first = false;
        }
        Ok(())
    }
}

impl<Data, Scalar> Zero for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    fn zero() -> Self {
        Self { data: BTreeMap::new() }
    }

    fn is_zero(&self) -> bool {
        self.data.is_empty()
    }
}

impl<Data, Scalar> Pow<u8> for &Polynomial<Monomial<Data>, Scalar>
where
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    for<'a> &'a Polynomial<Monomial<Data>, Scalar>:
        Mul<&'a Polynomial<Monomial<Data>, Scalar>, Output = Result<Polynomial<Monomial<Data>, Scalar>, String>>,
    Scalar: PolynomialDtype,
    Data: Ord + Clone + HasAMomentMatrixId,
    Monomial<Data>: OneWithMomentMatrixId,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn pow(self, rhs: u8) -> Self::Output {
        if self.is_zero() {
            Ok(self.clone())
        } else if let Some(mm_index) = self.get_unique_moment_matrix_id() {
            fallible_exponentiation_by_squaring(
                self,
                Polynomial::from(<Monomial<Data> as OneWithMomentMatrixId>::one(mm_index)),
                rhs as usize,
            )
        } else {
            Err("Can't compute the power of a polynomial having monomials with different moment matrix indices."
                .to_string())
        }
    }
}

impl<Id, Data, Scalar> From<&Operator<Id>> for Polynomial<Monomial<Data>, Scalar>
where
    for<'a> Monomial<Data>: From<&'a Operator<Id>>,
    Scalar: PolynomialDtype,
    Data: Ord,
{
    fn from(item: &Operator<Id>) -> Self {
        Self { data: BTreeMap::from([(Monomial::from(item), Scalar::one())]) }
    }
}

impl<Id, Data, Scalar> From<Operator<Id>> for Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: From<Operator<Id>>,
    Scalar: PolynomialDtype,
    Data: Ord,
{
    fn from(item: Operator<Id>) -> Self {
        Self { data: BTreeMap::from([(Monomial::from(item), Scalar::one())]) }
    }
}

impl<Data, Scalar> From<&Monomial<Data>> for Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord,
{
    fn from(item: &Monomial<Data>) -> Self {
        Self { data: BTreeMap::from([(item.clone(), Scalar::one())]) }
    }
}

impl<Data, Scalar> From<Monomial<Data>> for Polynomial<Monomial<Data>, Scalar>
where
    Scalar: PolynomialDtype,
    Data: Ord,
{
    fn from(item: Monomial<Data>) -> Self {
        Self { data: BTreeMap::from([(item, Scalar::one())]) }
    }
}

impl<Data, Scalar> Mul<Scalar> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn mul(self, rhs: Scalar) -> Self::Output {
        if rhs.is_zero() {
            Polynomial::zero()
        } else {
            Polynomial { data: self.data.iter().map(|(monomial, &coeff)| (monomial.clone(), coeff * rhs)).collect() }
        }
    }
}

impl<Data, Scalar> Mul<Scalar> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn mul(mut self, rhs: Scalar) -> Self::Output {
        if rhs.is_zero() {
            Polynomial::zero()
        } else {
            for coeff in self.data.values_mut() {
                *coeff = *coeff * rhs;
            }
            self
        }
    }
}

impl<Data, Scalar, Multiplier> Mul<&Multiplier> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
    Multiplier: HasAMomentMatrixId + Display,
    for<'a> &'a Monomial<Data>: Mul<&'a Multiplier, Output = Result<Monomial<Data>, String>>,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: &Multiplier) -> Self::Output {
        if self.is_zero() {
            Ok(Polynomial::zero())
        } else if let Some(mm_index) = self.get_unique_moment_matrix_id() {
            if mm_index != rhs.moment_matrix_id() {
                Err(format!(
                    "Cannot multiply polynomial {} with monomials having moment matrix index {} with operator {} having moment matrix index {}.",
                    self,
                    mm_index,
                    rhs,
                    rhs.moment_matrix_id()
                ))
            } else {
                let mut res = BTreeMap::new();

                for (mon, &coeff) in self.data.iter() {
                    manage_entry(&mut res, (mon * rhs)?, coeff);
                }

                Ok(Polynomial { data: res })
            }
        } else {
            Err("Cannot multiply a polynomial with monomials having different moment matrix indices with an operator."
                .to_string())
        }
    }
}

impl<Data, Scalar, Multiplier> Mul<&Multiplier> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
    Multiplier: HasAMomentMatrixId + Display,
    for<'a> Monomial<Data>: Mul<&'a Multiplier, Output = Result<Monomial<Data>, String>>,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: &Multiplier) -> Self::Output {
        if self.is_zero() {
            Ok(Polynomial::zero())
        } else if let Some(mm_index) = self.get_unique_moment_matrix_id() {
            if mm_index != rhs.moment_matrix_id() {
                Err(format!(
                    "Cannot multiply polynomial {} with monomials having moment matrix index {} with operator {} having moment matrix index {}.",
                    self,
                    mm_index,
                    rhs,
                    rhs.moment_matrix_id()
                ))
            } else {
                let mut res = BTreeMap::new();

                for (mon, coeff) in self.data.into_iter() {
                    manage_entry(&mut res, (mon * rhs)?, coeff);
                }

                Ok(Polynomial { data: res })
            }
        } else {
            Err("Cannot multiply a polynomial with monomials having different moment matrix indices with an operator."
                .to_string())
        }
    }
}

impl<Data, Scalar> Mul<&Polynomial<Monomial<Data>, Scalar>> for &Polynomial<Monomial<Data>, Scalar>
where
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Self::Output {
        if self.is_zero() | rhs.is_zero() {
            Ok(Polynomial::zero())
        } else if let (Some(mm_index_self), Some(mm_index_rhs)) =
            (self.get_unique_moment_matrix_id(), rhs.get_unique_moment_matrix_id())
        {
            if mm_index_self == mm_index_rhs {
                Ok(Polynomial { data: poly_mul(&self.data, &rhs.data)? })
            } else {
                Err(format!(
                    "Can't multiply polynomial {} with monomials having moment matrix index {} with polynomial {} with 
                    monomials having moment matrix index {}",
                    self, mm_index_self, rhs, mm_index_rhs
                ))
            }
        } else {
            Err("Can't multiply a polynomial having monomials with different moment matrix indices with another 
                polynomial."
                .to_string())
        }
    }
}

impl<Data, Scalar> Mul<&Polynomial<Monomial<Data>, Scalar>> for Polynomial<Monomial<Data>, Scalar>
where
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Self::Output {
        &self * rhs
    }
}

impl<Data, Scalar> Mul<Polynomial<Monomial<Data>, Scalar>> for &Polynomial<Monomial<Data>, Scalar>
where
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: Polynomial<Monomial<Data>, Scalar>) -> Self::Output {
        self * &rhs
    }
}

impl<Data, Scalar> Mul<Polynomial<Monomial<Data>, Scalar>> for Polynomial<Monomial<Data>, Scalar>
where
    for<'a> &'a Monomial<Data>: Mul<&'a Monomial<Data>, Output = Result<Monomial<Data>, String>>,
    Data: Ord + Clone,
    Monomial<Data>: HasAMomentMatrixId,
    Scalar: PolynomialDtype,
    Polynomial<Monomial<Data>, Scalar>: Display,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;

    fn mul(self, rhs: Polynomial<Monomial<Data>, Scalar>) -> Self::Output {
        &self * &rhs
    }
}

impl<Data, Scalar> Add<Scalar> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Clone + Ord,
    Scalar: PolynomialDtype,
    Monomial<Data>: OneWithMomentMatrixId,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;
    fn add(self, rhs: Scalar) -> Self::Output {
        if rhs.is_zero() {
            Ok(self.clone())
        } else if self.is_zero() {
            warn!("{} was added to a zero polynomial, its moment matrix index has been set to 0.", rhs);
            Ok(Monomial::one(0) * rhs)
        } else if let Some(mm_index) = self.get_unique_moment_matrix_id() {
            let mut self_data = self.data.clone();
            manage_entry(&mut self_data, <Monomial<Data> as OneWithMomentMatrixId>::one(mm_index), rhs);
            Ok(Polynomial { data: self_data })
        } else {
            Err("Adding a scalar to a polynomial with monomials having different moment matrix indices is ambiguous. 
                Use the dedicated identity operator instead."
                .to_string())
        }
    }
}

impl<Data, Scalar> Add<Scalar> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Clone + Ord,
    Scalar: PolynomialDtype,
    Monomial<Data>: OneWithMomentMatrixId,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;
    fn add(self, rhs: Scalar) -> Self::Output {
        if rhs.is_zero() {
            Ok(self.clone())
        } else if self.is_zero() {
            warn!("{} was added to a zero polynomial, its moment matrix index has been set to 0.", rhs);
            Ok(Monomial::one(0) * rhs)
        } else if let Some(mm_index) = self.get_unique_moment_matrix_id() {
            let mut self_data = self.data;
            manage_entry(&mut self_data, <Monomial<Data> as OneWithMomentMatrixId>::one(mm_index), rhs);
            Ok(Polynomial { data: self_data })
        } else {
            Err("Adding a scalar to a polynomial with monomials having different moment matrix indices is ambiguous. 
                Use the dedicated identity operator instead."
                .to_string())
        }
    }
}

impl<Data, Scalar> Sub<Scalar> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Clone + Ord,
    Scalar: PolynomialDtype,
    Monomial<Data>: OneWithMomentMatrixId,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;
    fn sub(self, rhs: Scalar) -> Self::Output {
        self + -rhs
    }
}

impl<Data, Scalar> Sub<Scalar> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Clone + Ord,
    Scalar: PolynomialDtype,
    Monomial<Data>: OneWithMomentMatrixId,
{
    type Output = Result<Polynomial<Monomial<Data>, Scalar>, String>;
    fn sub(self, rhs: Scalar) -> Self::Output {
        self + -rhs
    }
}

impl<Data, Scalar, Identifier> Add<&Operator<Identifier>> for &Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
    for<'a> Monomial<Data>: From<&'a Operator<Identifier>>,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: &Operator<Identifier>) -> Polynomial<Monomial<Data>, Scalar> {
        self + Polynomial::from(rhs)
    }
}

impl<Data, Scalar> Add<&Monomial<Data>> for &Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: &Monomial<Data>) -> Polynomial<Monomial<Data>, Scalar> {
        self + Polynomial::from(rhs)
    }
}

impl<Data, Scalar> Add<&Monomial<Data>> for Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: &Monomial<Data>) -> Polynomial<Monomial<Data>, Scalar> {
        self + Polynomial::from(rhs)
    }
}

impl<Data, Scalar> Add<Monomial<Data>> for Polynomial<Monomial<Data>, Scalar>
where
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: Monomial<Data>) -> Polynomial<Monomial<Data>, Scalar> {
        self + Polynomial::from(rhs)
    }
}

impl<Data, Scalar, Identifier> Sub<&Operator<Identifier>> for &Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
    for<'a> Monomial<Data>: From<&'a Operator<Identifier>>,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: &Operator<Identifier>) -> Polynomial<Monomial<Data>, Scalar> {
        self - Polynomial::from(rhs)
    }
}

impl<Data, Scalar> Sub<&Monomial<Data>> for &Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: &Monomial<Data>) -> Polynomial<Monomial<Data>, Scalar> {
        self - Polynomial::from(rhs)
    }
}

impl<Data, Scalar> Sub<&Monomial<Data>> for Polynomial<Monomial<Data>, Scalar>
where
    Monomial<Data>: Clone,
    Scalar: PolynomialDtype,
    Data: Ord + Clone,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: &Monomial<Data>) -> Polynomial<Monomial<Data>, Scalar> {
        self - Polynomial::from(rhs)
    }
}

impl<Data, Scalar> Add<&Polynomial<Monomial<Data>, Scalar>> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial {
            data: merge_btreemaps(&self.data, &rhs.data, |_, coeff_left, coeff_right| coeff_left + coeff_right),
        }
    }
}

impl<Data, Scalar> Sub<&Polynomial<Monomial<Data>, Scalar>> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial {
            data: merge_btreemaps(&self.data, &rhs.data, |_, coeff_left, coeff_right| coeff_left - coeff_right),
        }
    }
}

impl<Data, Scalar> Add<&Polynomial<Monomial<Data>, Scalar>> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial {
            data: merge_btreemaps(self.data, &rhs.data, |_, coeff_left, coeff_right| coeff_left + coeff_right),
        }
    }
}

impl<Data, Scalar> Sub<&Polynomial<Monomial<Data>, Scalar>> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: &Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial {
            data: merge_btreemaps(self.data, &rhs.data, |_, coeff_left, coeff_right| coeff_left - coeff_right),
        }
    }
}

impl<Data, Scalar> Add<Polynomial<Monomial<Data>, Scalar>> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial {
            data: merge_btreemaps(&self.data, rhs.data, |_, coeff_left, coeff_right| coeff_left + coeff_right),
        }
    }
}

impl<Data, Scalar> Sub<Polynomial<Monomial<Data>, Scalar>> for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial {
            data: merge_btreemaps(&self.data, rhs.data, |_, coeff_left, coeff_right| coeff_left - coeff_right),
        }
    }
}

impl<Data, Scalar> Add<Polynomial<Monomial<Data>, Scalar>> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn add(self, rhs: Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial { data: merge_btreemaps(self.data, rhs.data, |_, coeff_left, coeff_right| coeff_left + coeff_right) }
    }
}

impl<Data, Scalar> Sub<Polynomial<Monomial<Data>, Scalar>> for Polynomial<Monomial<Data>, Scalar>
where
    Data: Ord + Clone,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn sub(self, rhs: Polynomial<Monomial<Data>, Scalar>) -> Polynomial<Monomial<Data>, Scalar> {
        Polynomial { data: merge_btreemaps(self.data, rhs.data, |_, coeff_left, coeff_right| coeff_left - coeff_right) }
    }
}

impl<Data, Scalar> Neg for &Polynomial<Monomial<Data>, Scalar>
where
    Data: Clone + Ord,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn neg(self) -> Self::Output {
        Polynomial { data: self.data.iter().map(|(monom, &coeff)| (monom.clone(), -coeff)).collect() }
    }
}

impl<Data, Scalar> Neg for Polynomial<Monomial<Data>, Scalar>
where
    Data: Clone + Ord,
    Scalar: PolynomialDtype,
{
    type Output = Polynomial<Monomial<Data>, Scalar>;

    fn neg(mut self) -> Self::Output {
        for coeff in self.data.values_mut() {
            *coeff = -*coeff;
        }
        self
    }
}

// TODO: check that the following tests are not already covered somewhere else
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use num_complex::Complex;
    use num_traits::Zero;

    use crate::polynomials::monomial::OneWithMomentMatrixId;
    use crate::polynomials::noncommutative_polynomials::monomials::noncommutative_monomial::RustNonCommutativeMonomial;
    use crate::polynomials::noncommutative_polynomials::operators::noncommutative_operator::RustNonCommutativeOperator;
    use crate::polynomials::noncommutative_polynomials::polynomials::noncommutative_polynomial::RustComplexCoefficientsNonCommutativePolynomial;

    #[test]
    fn test_add_operator() {
        let op = RustNonCommutativeOperator::new('x', 0, false, false, false, 0);
        let poly = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([(RustNonCommutativeMonomial::new(vec![op], 0), Complex::ONE)]),
        };
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([(RustNonCommutativeMonomial::new(vec![op], 0), Complex { re: 2.0, im: 0.0 })]),
        };
        assert_eq!(&poly + &op, expected);
    }

    #[test]
    fn test_sub_operator() {
        let op = RustNonCommutativeOperator::new('x', 0, false, false, false, 0);
        let poly = RustComplexCoefficientsNonCommutativePolynomial {
            data: BTreeMap::from([(RustNonCommutativeMonomial::new(vec![op], 0), Complex::ONE)]),
        };
        assert_eq!(&poly - &op, RustComplexCoefficientsNonCommutativePolynomial::zero());
    }

    #[test]
    fn test_add_monomial() {
        let z = RustNonCommutativeOperator::new('z', 0, false, false, false, 0);
        let new_mon = RustNonCommutativeMonomial::new(vec![z], 0);
        let poly = RustComplexCoefficientsNonCommutativePolynomial {
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
            ]),
        };
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
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
                (new_mon.clone(), Complex::ONE),
            ]),
        };
        assert_eq!(&poly + &new_mon, expected);
    }

    #[test]
    fn test_sub_monomial_new_entry() {
        let z = RustNonCommutativeOperator::new('z', 0, false, false, false, 0);
        let new_mon = RustNonCommutativeMonomial::new(vec![z], 0);
        let poly = RustComplexCoefficientsNonCommutativePolynomial {
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
            ]),
        };
        let expected = RustComplexCoefficientsNonCommutativePolynomial {
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
                (new_mon.clone(), -Complex::ONE),
            ]),
        };
        assert_eq!(&poly - &new_mon, expected);
    }
}
