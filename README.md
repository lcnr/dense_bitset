# Dense Bitset

A dense bitset implemented in `rust` using only safe code.

## Examples

```rust
use dense_bitset::BitSet;

let mut set = BitSet::new();

set.insert(7);
set.set(4, true);
set.flip(5);

assert_eq!(set, [7, 4, 5].iter().collect());

set.remove(7);
set.flip(4);
set.set(5, false);

assert!(set.is_empty());

let a: BitSet = [2, 5, 12, 17].iter().collect();
let b: BitSet = [2, 12].iter().collect();

assert!(!a.is_disjoint(&b));
assert!(b.is_subset(&a));
assert!(a.is_superset(&b));
```
