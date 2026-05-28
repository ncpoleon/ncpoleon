use std::ops::Mul;

pub(crate) fn fallible_exponentiation_by_squaring<T>(x: &T, identity: T, mut power: usize) -> Result<T, String>
where
    T: Clone,
    for<'a> &'a T: Mul<&'a T, Output = Result<T, String>>,
{
    if power == 0 {
        return Ok(identity);
    }
    let mut base = x.clone();
    let mut acc = identity;

    // Iterative version of exponentiation by squaring
    while power > 1 {
        if power & 1 == 1 {
            acc = (&acc * &base)?;
            power -= 1;
        }
        base = (&base * &base)?;
        power >>= 1;
    }

    &base * &acc
}
