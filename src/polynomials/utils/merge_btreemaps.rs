use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::btree_map::{IntoIter, Iter};
use std::iter::Map;

use log::trace;
use num_traits::Zero;

pub(crate) trait IntoIterOfOwned<U, V> {
    type IterType: Iterator<Item = (U, V)>;
    fn into_iter_of_owned(self) -> Self::IterType;
}

impl<U, V> IntoIterOfOwned<U, V> for BTreeMap<U, V> {
    type IterType = IntoIter<U, V>;

    fn into_iter_of_owned(self) -> Self::IterType {
        self.into_iter()
    }
}

impl<'a, U, V> IntoIterOfOwned<U, V> for &'a BTreeMap<U, V>
where
    U: Clone,
    V: Copy,
{
    type IterType = Map<Iter<'a, U, V>, fn((&U, &V)) -> (U, V)>;

    fn into_iter_of_owned(self) -> Self::IterType {
        self.iter().map(|(key, &value)| (key.clone(), value))
    }
}

/// Merge two BTreeMaps according to an aggregator function.
pub(crate) fn merge_btreemaps<U, V, L, R, F>(left: L, right: R, aggregator: F) -> BTreeMap<U, V>
where
    U: Ord,
    V: Zero,
    L: IntoIterOfOwned<U, V>,
    R: IntoIterOfOwned<U, V>,
    // The aggregator takes a reference to the insert key as input if further normalization is
    // needed, for instance setting the power to 1 when multiplying two commutative monomials
    // containing the same projector
    F: Fn(&U, V, V) -> V,
{
    let mut res = BTreeMap::new();
    let mut left_iter = left.into_iter_of_owned();
    let mut right_iter = right.into_iter_of_owned();

    let mut left_elt = left_iter.next();
    let mut right_elt = right_iter.next();

    loop {
        let left_elt_owned = left_elt.take();
        let right_elt_owned = right_elt.take();

        // match moves the variable, so since we want to own the variable so that we don't re-clone
        // the key, we re-create a variable each time
        match (left_elt_owned, right_elt_owned) {
            (Some((left_key, left_value)), Some((right_key, right_value))) => {
                let insert_key;
                let new_value;

                match left_key.cmp(&right_key) {
                    Ordering::Less => {
                        new_value = aggregator(&left_key, left_value, V::zero());
                        insert_key = left_key;
                        left_elt = left_iter.next();
                        right_elt = Some((right_key, right_value));
                    }
                    Ordering::Equal => {
                        trace!("Key collision in merge_btreemaps, aggregating values.");
                        new_value = aggregator(&left_key, left_value, right_value);
                        insert_key = left_key;
                        left_elt = left_iter.next();
                        right_elt = right_iter.next();
                    }
                    Ordering::Greater => {
                        new_value = aggregator(&right_key, V::zero(), right_value);
                        insert_key = right_key;
                        right_elt = right_iter.next();
                        left_elt = Some((left_key, left_value))
                    }
                }

                if !new_value.is_zero() {
                    res.insert(insert_key, new_value);
                }
            }
            (Some((left_key, left_value)), None) => {
                for (key, value) in std::iter::once((left_key, left_value)).chain(left_iter) {
                    let new_value = aggregator(&key, value, V::zero());
                    if !new_value.is_zero() {
                        res.insert(key, new_value);
                    }
                }
                break;
            }
            (None, Some((right_key, right_value))) => {
                for (key, value) in std::iter::once((right_key, right_value)).chain(right_iter) {
                    let new_value = aggregator(&key, V::zero(), value);
                    if !new_value.is_zero() {
                        res.insert(key, new_value);
                    }
                }
                break;
            }
            (None, None) => break,
        }
    }

    res
}
