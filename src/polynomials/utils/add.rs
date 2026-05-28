use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ops::Add;

use num_traits::Zero;

/// Convenience function to insert or remove an element in a BTreeMap when inserting another
/// element.
///
/// It first checks whether the key is already present. If it's not the case, it inserts the
/// provided value and exits. Otherwise, it checks whether the resulting value is nil, in which case
/// it removes the entry. If it's not, it updates the value.
pub(crate) fn manage_entry<U, V>(res: &mut BTreeMap<U, V>, monom: U, coeff: V)
where
    U: Ord + Clone,
    V: Copy + Zero + Add<V, Output = V>,
{
    match res.entry(monom) {
        Entry::Vacant(entry) => {
            entry.insert(coeff);
        }
        Entry::Occupied(mut entry) => {
            let new_value = *entry.get() + coeff;
            if new_value.is_zero() {
                entry.remove();
            } else {
                *entry.get_mut() = new_value;
            }
        }
    }
}
