//! A dense bitset implementation using only safe code.
//!
//! # Examples
//!
//! ```
//! use dense_bitset::BitSet;
//!
//! let mut set = BitSet::new();
//!
//! set.insert(5);
//! set.insert(42);
//! set.insert(7);
//! assert!(set.get(5));
//!
//! assert_eq!(set, [5, 7, 42].iter().collect());
//! ```
#[forbid(unsafe_code)]
use std::{
    borrow::Borrow,
    cmp::{Eq, PartialEq},
    fmt::{self, Write},
    iter::{DoubleEndedIterator, Extend, FromIterator, FusedIterator},
    mem,
};

type Frame = u64;

const FRAME_SIZE: usize = mem::size_of::<Frame>() * 8;

/// A variably sized, heap allocated, dense bitset implemented using no `unsafe` code.
///
/// # Examples
///
/// ```
/// use dense_bitset::BitSet;
///
/// let mut set = BitSet::new();
///
/// set.insert(7);
/// set.set(4, true);
/// set.flip(5);
///
/// assert_eq!(set, [7, 4, 5].iter().collect());
///
/// set.remove(7);
/// set.flip(4);
/// set.set(5, false);
///
/// assert!(set.is_empty());
///
/// let a: BitSet = [2, 5, 12, 17].iter().collect();
/// let b: BitSet = [2, 12].iter().collect();
///
/// assert!(!a.is_disjoint(&b));
/// assert!(b.is_subset(&a));
/// assert!(a.is_superset(&b));
/// ```
pub struct BitSet {
    inner: Vec<Frame>,
}

impl fmt::Debug for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct SFmt<'a>(&'a str);

        impl<'a> fmt::Debug for SFmt<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        let mut temp = String::new();
        for frame in self.inner.iter() {
            write!(temp, "{:01$b}", frame.reverse_bits(), FRAME_SIZE)?;
        }

        let mut r = temp.trim_end_matches('0');
        if r.is_empty() {
            r = "0";
        }

        f.debug_struct("BitSet").field("inner", &SFmt(r)).finish()
    }
}

impl Clone for BitSet {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
    }
}

impl PartialEq for BitSet {
    fn eq(&self, rhs: &Self) -> bool {
        if self.inner.len() != rhs.inner.len() {
            false
        } else {
            self.inner.iter().zip(rhs.inner.iter()).all(|(a, b)| a == b)
        }
    }
}

impl Eq for BitSet {}

impl BitSet {
    /// Constructs a new, empty `BitSet`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    ///
    /// set.insert(7);
    /// assert_eq!(1, set.element_count());
    /// ```
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Constructs a new, empty `BitSet` with at least the specified capacity.
    /// All indices which are smaller than this capacity,
    /// can be used without requiring a reallocation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::with_capacity(100);
    /// let capacity = set.capacity();
    /// assert!(capacity >= 100);
    /// set.insert(99);
    /// assert_eq!(capacity, set.capacity());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        let frames = if capacity % FRAME_SIZE == 0 {
            capacity / FRAME_SIZE
        } else {
            capacity / FRAME_SIZE + 1
        };

        Self {
            inner: Vec::with_capacity(frames),
        }
    }

    /// Removes all trailing frames containing `0`.
    #[inline]
    fn remove_empty_frames(&mut self) {
        while self.inner.last().map_or(false, |&l| l == 0) {
            self.inner.pop();
        }
    }

    /// Returns the capacity of this `BitSet`.
    /// All indices which are smaller than this capacity,
    /// can be used without requiring a reallocation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::with_capacity(100);
    /// assert!(set.capacity() >= 100);
    /// ```
    pub fn capacity(&self) -> usize {
        self.inner.capacity() * FRAME_SIZE
    }

    /// Returns `true` if no entries are set to `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    /// assert!(set.is_empty());
    ///
    /// set.insert(99);
    /// assert!(!set.is_empty());
    ///
    /// set.remove(99);
    /// assert!(set.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner.iter().all(|&frame| frame == 0)
    }

    /// Returns the amount of entries set to `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    /// assert_eq!(0, set.element_count());
    ///
    /// set.insert(1729);
    /// set.insert(1337);
    /// assert_eq!(2, set.element_count());
    /// ```
    pub fn element_count(&self) -> usize {
        self.inner
            .iter()
            .fold(0, |sum, elem| sum + elem.count_ones() as usize)
    }

    /// Shrinks the `capacity` as much as possible.
    ///
    /// While the `capacity` will drop down as close as possible to the biggest set `idx`,
    /// there might still be space for a few more elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::with_capacity(1000);
    /// set.extend([100, 200, 300].iter());
    /// assert!(set.capacity() >= 1000);
    ///
    /// set.shrink_to_fit();
    /// assert!(set.capacity() > 300);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Sets the entry at `idx` to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    ///
    /// set.set(1337, true);
    /// assert_eq!(true, set.get(1337));
    ///
    /// set.set(1337, false);
    /// assert_eq!(false, set.get(1337));
    /// ```
    pub fn set(&mut self, idx: usize, value: bool) {
        if value {
            self.insert(idx)
        } else {
            self.remove(idx)
        }
    }

    /// Sets the entry at `idx` to `true`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    ///
    /// set.insert(69);
    /// assert_eq!(true, set.get(69));
    ///
    /// set.remove(69);
    /// assert_eq!(false, set.get(69));
    /// ```
    pub fn insert(&mut self, idx: usize) {
        let frame_offset = idx / FRAME_SIZE;
        if frame_offset >= self.inner.len() {
            self.inner.resize(frame_offset + 1, 0);
        }

        self.inner[frame_offset] |= 1 << idx - frame_offset * FRAME_SIZE;
    }

    /// Sets the entry at `idx` to `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    ///
    /// set.insert(42);
    /// assert_eq!(true, set.get(42));
    ///
    /// set.remove(42);
    /// assert_eq!(false, set.get(42));
    /// ```
    pub fn remove(&mut self, idx: usize) {
        let frame_offset = idx / FRAME_SIZE;
        if frame_offset < self.inner.len() {
            self.inner[frame_offset] &= !(1 << idx - frame_offset * FRAME_SIZE);
        }

        self.remove_empty_frames();
    }

    /// Inverts the value of the entry at `idx`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    /// assert_eq!(false, set.get(42));
    ///
    /// set.flip(42);
    /// assert_eq!(true, set.get(42));
    ///
    /// set.flip(42);
    /// assert_eq!(false, set.get(42));
    /// ```
    pub fn flip(&mut self, idx: usize) {
        let frame_offset = idx / FRAME_SIZE;
        if frame_offset >= self.inner.len() {
            self.inner.resize(frame_offset + 1, 0);
        }

        self.inner[frame_offset] ^= 1 << idx - frame_offset * FRAME_SIZE;
        self.remove_empty_frames();
    }

    /// Returns the value of the entry at `idx`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    /// assert_eq!(false, set.get(7));
    ///
    /// set.insert(6);
    /// assert_eq!(false, set.get(7));
    ///
    /// set.insert(7);
    /// assert_eq!(true, set.get(7));
    /// ```
    pub fn get(&self, idx: usize) -> bool {
        let frame_offset = idx / FRAME_SIZE;
        self.inner
            .get(frame_offset)
            .map_or(false, |v| v & (1 << idx - frame_offset * FRAME_SIZE) != 0)
    }

    /// Returns if `self` and `other` do not share a `true` element with the same index.
    /// This is equivalent to checking for an empty intersection.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let a: BitSet = [1, 2, 3].iter().collect();
    /// let mut b = BitSet::new();
    ///
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(4);
    /// assert_eq!(a.is_disjoint(&b), true);
    /// b.insert(1);
    /// assert_eq!(a.is_disjoint(&b), false);
    /// ```
    pub fn is_disjoint(&self, other: &BitSet) -> bool {
        self.inner
            .iter()
            .zip(other.inner.iter())
            .all(|(a, b)| a & b == 0)
    }

    /// Returns `true` if the set is a subset of `other`,
    /// meaning `other` contains at least all the values in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let sup: BitSet = [1, 2, 3].iter().collect();
    /// let mut set = BitSet::new();
    ///
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(2);
    /// assert_eq!(set.is_subset(&sup), true);
    /// set.insert(4);
    /// assert_eq!(set.is_subset(&sup), false);
    /// ```
    pub fn is_subset(&self, other: &BitSet) -> bool {
        if self.inner.len() <= other.inner.len() {
            self.inner
                .iter()
                .zip(other.inner.iter())
                .all(|(a, b)| a & !b == 0)
        } else {
            false
        }
    }

    /// Returns `true` if `self` is a superset of `other`,
    /// meaning `self` contains at least all the values in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let sup: BitSet = [1, 2].iter().collect();
    /// let mut set = BitSet::new();
    ///
    /// assert_eq!(set.is_superset(&sup), false);
    ///
    /// set.insert(0);
    /// set.insert(1);
    /// assert_eq!(set.is_superset(&sup), false);
    ///
    /// set.insert(2);
    /// assert_eq!(set.is_superset(&sup), true);
    /// ```
    #[inline]
    pub fn is_superset(&self, other: &BitSet) -> bool {
        other.is_subset(self)
    }

    /// Returns the highest index for which the entry which is set to `true`.
    /// In case there is no `true` entry, this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    /// assert_eq!(None, set.highest_bit());
    ///
    /// set.insert(3);
    /// set.insert(7);
    /// set.insert(4);
    /// assert_eq!(Some(7), set.highest_bit());
    /// ```
    pub fn highest_bit(&self) -> Option<usize> {
        self.iter().next_back()
    }

    /// Returns the lowest index for which the entry is set to `true`.
    /// In case there is no `true` entry, this method returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use dense_bitset::BitSet;
    /// let mut set = BitSet::new();
    /// assert_eq!(None, set.lowest_bit());
    ///
    /// set.insert(3);
    /// set.insert(7);
    /// set.insert(4);
    /// assert_eq!(Some(3), set.lowest_bit());
    /// ```
    pub fn lowest_bit(&self) -> Option<usize> {
        self.iter().next()
    }

    /// Returns an iterator over the bitset which returns all indices of entries set to `true`.
    /// The indices are sorted from lowest to highest.
    ///
    /// # Examples
    ///
    /// ```
    /// # use dense_bitset::BitSet;
    /// let set: BitSet = [1, 12, 19, 4].iter().copied().collect();
    /// let mut iter = set.iter();
    ///
    /// assert_eq!(Some(1), iter.next());
    /// assert_eq!(Some(4), iter.next());
    /// assert_eq!(Some(12), iter.next());
    /// assert_eq!(Some(19), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn iter(&self) -> IdxIter<&Self> {
        IdxIter {
            inner: self,
            pos: 0,
            end_pos: self.inner.len() * FRAME_SIZE,
        }
    }
}

impl<'a> Extend<&'a usize> for BitSet {
    fn extend<I: IntoIterator<Item = &'a usize>>(&mut self, iter: I) {
        for item in iter {
            self.insert(*item);
        }
    }
}

impl Extend<usize> for BitSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<'a> FromIterator<&'a bool> for BitSet {
    #[inline]
    fn from_iter<U: IntoIterator<Item = &'a bool>>(iter: U) -> BitSet {
        Self::from_iter(iter.into_iter().copied())
    }
}

impl FromIterator<bool> for BitSet {
    fn from_iter<U: IntoIterator<Item = bool>>(iter: U) -> BitSet {
        let mut set = BitSet::new();
        for (idx, value) in iter.into_iter().enumerate() {
            if value {
                set.insert(idx);
            }
        }

        set
    }
}

impl<'a> FromIterator<&'a usize> for BitSet {
    #[inline]
    fn from_iter<U: IntoIterator<Item = &'a usize>>(iter: U) -> BitSet {
        Self::from_iter(iter.into_iter().copied())
    }
}

impl FromIterator<usize> for BitSet {
    fn from_iter<U: IntoIterator<Item = usize>>(iter: U) -> BitSet {
        let mut set = BitSet::new();
        for idx in iter {
            set.insert(idx);
        }
        set
    }
}

impl IntoIterator for BitSet {
    type Item = usize;
    type IntoIter = IdxIter<BitSet>;

    fn into_iter(self) -> IdxIter<BitSet> {
        let end_pos = self.inner.len() * FRAME_SIZE;

        IdxIter {
            inner: self,
            pos: 0,
            end_pos,
        }
    }
}

/// A iterator over the `true` entries of a `BitSet`.
///
/// This struct is created by calling [`BitSet::iter`].
///
/// # Examples
///
/// ```
/// # use dense_bitset::BitSet;
/// use dense_bitset::IdxIter;
///
/// let set: BitSet = [4, 3, 12, 19].iter().collect();
///
/// let mut ref_iter = set.iter();
/// assert!([3, 4, 12, 19].iter().all(|&e| e == ref_iter.next().unwrap()));
/// assert_eq!(None, ref_iter.next());
///
/// let mut owned_iter = set.into_iter();
/// assert!([3, 4, 12, 19].iter().all(|&e| e == owned_iter.next().unwrap()));
/// assert_eq!(None, owned_iter.next());
/// ```
/// [`BitSet::iter`]: ./struct.BitSet.html#method.iter
pub struct IdxIter<B> {
    inner: B,
    pos: usize,
    end_pos: usize,
}

impl<B: Borrow<BitSet>> Iterator for IdxIter<B> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.pos <= self.end_pos {
            let pos = self.pos;
            self.pos += 1;
            if self.inner.borrow().get(pos) {
                return Some(pos);
            }
        }
        None
    }
}

impl<B: Borrow<BitSet>> DoubleEndedIterator for IdxIter<B> {
    fn next_back(&mut self) -> Option<usize> {
        while self.end_pos > self.pos {
            let pos = self.end_pos;
            self.end_pos -= 1;
            if self.inner.borrow().get(pos) {
                return Some(pos);
            }
        }

        if self.end_pos == self.pos {
            self.pos += 1;
            if self.inner.borrow().get(self.end_pos) {
                return Some(self.end_pos);
            }
        }

        None
    }
}

impl<B: Borrow<BitSet>> FusedIterator for IdxIter<B> {}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter;

    #[test]
    fn with_capacity() {
        for cap in 0..FRAME_SIZE * 2 {
            let mut set = BitSet::with_capacity(cap);

            let frames = set.capacity();
            for i in 0..cap {
                set.insert(i);
                assert_eq!(frames, set.capacity(), "{}/{}", i, cap);
            }
        }
    }

    #[test]
    fn test() {
        let mut set = BitSet::new();
        assert_eq!(set.element_count(), 0);
        assert_eq!(set.get(1000000), false);
        assert_eq!(set.inner.len(), 0);
        assert!(set.is_empty());
        set.insert(3);
        assert_eq!(set.inner.len(), 1);
        assert_eq!(set.get(3), true);
        assert_eq!(set.get(4), false);
        set.insert(5);
        assert_eq!(set.element_count(), 2);
        assert!(!set.is_empty());
        assert_eq!(set.get(5), true);
        set.insert(FRAME_SIZE + 2);
        assert_eq!(set.inner.len(), 2);
        assert_eq!(set.get(FRAME_SIZE + 2), true);
        assert_eq!(set.get(FRAME_SIZE + 1), false);
        set.flip(FRAME_SIZE + 4);
        assert_eq!(set.get(FRAME_SIZE), false);
        assert_eq!(set.get(FRAME_SIZE + 2), true);
        assert_eq!(set.get(FRAME_SIZE + 4), true);
        set.flip(FRAME_SIZE + 4);
        assert_eq!(set.get(FRAME_SIZE + 4), false);
        set.flip(FRAME_SIZE * 2 + 1);
        assert_eq!(set.inner.len(), 3);
        assert_eq!(set.get(FRAME_SIZE * 2 + 1), true);
        assert_eq!(set.get(FRAME_SIZE * 2 + 3), false);
        set.remove(FRAME_SIZE * 2 + 1);
        assert_eq!(set.get(FRAME_SIZE * 2 + 1), false);
        set.remove(FRAME_SIZE * 2 + 1);
        assert_eq!(set.get(FRAME_SIZE * 2 + 1), false);
        set.remove(FRAME_SIZE * 100);
        assert_eq!(set.inner.len(), 2);
        assert_eq!(set.element_count(), 3);
    }

    #[test]
    fn disjoint() {
        let a: BitSet = [1, 3].iter().collect();
        let b: BitSet = [2, 5].iter().collect();
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn eq() {
        let mut a = BitSet::new();
        let mut b = BitSet::new();
        a.insert(FRAME_SIZE * 2);
        assert_ne!(a, b);
        b.insert(FRAME_SIZE * 2);
        assert_eq!(a, b);
        a.insert(FRAME_SIZE * 3);
        assert_ne!(a, b);
        a.remove(FRAME_SIZE * 3);
        assert_eq!(a, b);
        b.insert(FRAME_SIZE * 4);
        assert_ne!(a, b);
        b.remove(FRAME_SIZE * 4);
        assert_eq!(a, b);
    }

    #[test]
    fn iter() {
        let mut set: BitSet = [7, 4, 3, 4, 1, 1000].iter().collect();
        assert_eq!(set.get(1), true);
        assert_eq!(set.get(2), false);
        assert_eq!(set.get(4), true);
        set.insert(0);
        assert_eq!(set.get(0), true);
        assert_eq!(set.get(7), true);
        assert_eq!(set.get(99), false);
        assert_eq!(set.get(1000), true);

        let mut iter = set.iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(1000));
        assert_eq!(iter.next(), None);

        set.remove(0);
        set.extend(iter::once(5).chain(iter::once(1)));
        assert_eq!(set.get(1), true);
        assert_eq!(set.get(2), false);
        assert_eq!(set.get(5), true);

        let mut iter = set.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next_back(), Some(1000));
        assert_eq!(iter.next_back(), Some(7));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        let set: BitSet = [0, 1].iter().collect();
        let mut iter = set.iter();
        assert_eq!(iter.next_back(), Some(1));
        assert_eq!(iter.next_back(), Some(0));
        assert_eq!(iter.next_back(), None);
        let mut iter = set.iter();
        assert_eq!(iter.next_back(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), None);
        let mut iter = set.iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next_back(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn debug() {
        fn assert_debug(content: &str, set: &BitSet) {
            assert_eq!(
                format!("BitSet {{ inner: {} }}", content),
                format!("{:?}", set)
            );
        }

        let mut set = BitSet::new();
        assert_debug("0", &set);

        set.insert(1);
        assert_debug("01", &set);

        set.insert(64);
        assert_debug(
            "01000000000000000000000000000000000000000000000000000000000000001",
            &set,
        );
    }
}
