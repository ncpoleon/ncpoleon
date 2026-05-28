use std::collections::BTreeMap;
use std::ops::Mul;

use crate::polynomials::monomial::Monomial;
use crate::polynomials::polynomial::PolynomialDtype;
use crate::polynomials::utils::add::manage_entry;

pub(crate) fn poly_mul<U, V>(
    data_left: &BTreeMap<Monomial<U>, V>,
    data_right: &BTreeMap<Monomial<U>, V>,
) -> Result<BTreeMap<Monomial<U>, V>, String>
where
    U: Clone + Ord,
    for<'a> &'a Monomial<U>: Mul<&'a Monomial<U>, Output = Result<Monomial<U>, String>>,
    V: PolynomialDtype,
{
    let mut res = BTreeMap::new();

    for (monom_left, &coeff_left) in data_left.iter() {
        for (monom_right, &coeff_right) in data_right.iter() {
            let new_coeff = coeff_left * coeff_right;
            manage_entry(&mut res, (monom_left * monom_right)?, new_coeff);
        }
    }

    Ok(res)
}
