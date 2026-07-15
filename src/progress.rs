use itertools::Either;
use kdam::{Bar, BarExt};

// TODO: we may want to make these configurable from Python, but for now, let's just hardcode them.

/// Number of elements consumed between two `Bar::update` calls.
///
/// `Bar::update` reads the clock unconditionally (`mininterval` only throttles the *rendering*, not
/// the bookkeeping), so wrapping a hot iterator with `tqdm!` costs a `clock_gettime` per element.
const STRIDE: usize = 1024;

/// Seconds between two renders of a progress bar.
///
/// Every render is a `write` (plus a `term::width` ioctl) on stderr. Might be slow on some terminals, so keep
/// reasonable
pub(crate) const MININTERVAL: f32 = 0.5;

/// Iterator adapter that advances `bar` once every [`STRIDE`] elements instead of once per element.
pub(crate) struct ThrottledBarIter<I> {
    inner: I,
    bar: Bar,
    pending: usize,
}

impl<I: Iterator> Iterator for ThrottledBarIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next();

        if item.is_some() {
            self.pending += 1;

            if self.pending >= STRIDE {
                let _ = self.bar.update(self.pending);
                self.pending = 0;
            }
        } else if self.pending > 0 {
            // Flush the remainder on exhaustion so the bar reaches its total.
            let _ = self.bar.update(self.pending);
            self.pending = 0;
        }

        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for ThrottledBarIter<I> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Attaches `bar` to `iter`, when there is one, without paying a per-element clock read.
///
/// Pass `None` to iterate without any progress bar; the iterator is then returned untouched, so a
/// run without progress bars is not taxed at all.
pub(crate) fn with_bar<I: Iterator>(iter: I, bar: Option<Bar>) -> Either<ThrottledBarIter<I>, I> {
    match bar {
        Some(bar) => Either::Left(ThrottledBarIter { inner: iter, bar, pending: 0 }),
        None => Either::Right(iter),
    }
}
