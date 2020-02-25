# Changelog

## master

- remove `IntoIterator` implementation from `BitSet`.
- rename `BitSet::iter` to `BitSet::indices`.
- rename `Iter` to `Indices` and remove owned variant.
- add `fn BitSet::entries`, an interator returning the value of each entry.

## 0.1.1

- implement `Default` for `BitSet`.

## 0.1.0

Initial release