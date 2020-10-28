//! This crate implements a lockfree cuckoo hashmap.
#![deny(
    // The following are allowed by default lints according to
    // https://doc.rust-lang.org/rustc/lints/listing/allowed-by-default.html
    anonymous_parameters,
    bare_trait_objects,
    // box_pointers, // futures involve boxed pointers
    elided_lifetimes_in_paths, // allow anonymous lifetime in generated code
    missing_copy_implementations,
    missing_debug_implementations,
    // missing_docs, // TODO: add documents
    single_use_lifetimes, // TODO: fix lifetime names only used once
    trivial_casts, // TODO: remove trivial casts in code
    trivial_numeric_casts,
    // unreachable_pub, use clippy::redundant_pub_crate instead
    // unsafe_code, unsafe codes are inevitable here
    unstable_features,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    // unused_results, // TODO: fix unused results
    variant_size_differences,

    // Treat warnings as errors
    warnings,

    clippy::all,
    clippy::restriction,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo
)]
#![allow(
    // Some explicitly allowed Clippy lints, must have clear reason to allow
    clippy::blanket_clippy_restriction_lints, // allow clippy::restriction
    clippy::panic, // allow debug_assert, panic in production code
    clippy::implicit_return, // actually omitting the return keyword is idiomatic Rust code
)]

/// `pointer` defines atomic pointers which will be used for lockfree operations.
mod pointer;

/// `map_inner` defines the inner implementation of the hashmap.
mod map_inner;

use map_inner::{KVPair, MapInner};

use std::collections::hash_map::RandomState;

use std::hash::Hash;

use std::sync::atomic::Ordering;

use pointer::{AtomicPtr, SharedPtr};

// Re-export `crossbeam_epoch::pin()` and `crossbeam_epoch::Guard`.
pub use crossbeam_epoch::{pin, Guard};

/// `LockFreeCuckooHash` is a lock-free hash table using cuckoo hashing scheme.
/// This implementation is based on the approach discussed in the paper:
///
/// "Nguyen, N., & Tsigas, P. (2014). Lock-Free Cuckoo Hashing. 2014 IEEE 34th International
/// Conference on Distributed Computing Systems, 627-636."
///
/// Cuckoo hashing is an open addressing solution for hash collisions. The basic idea of cuckoo
/// hashing is to resolve collisions by using two or more hash functions instead of only one. In this
/// implementation, we use two hash functions and two arrays (or tables).
///
/// The search operation only looks up two slots, i.e. table[0][hash0(key)] and table[1][hash1(key)].
/// If these two slots do not contain the key, the hash table does not contain the key. So the search operation
/// only takes a constant time in the worst case.
///
/// The insert operation must pay the price for the quick search. The insert operation can only put the key
/// into one of the two slots. However, when both slots are already occupied by other entries, it will be
/// necessary to move other keys to their second locations (or back to their first locations) to make room
/// for the new key, which is called a `relocation`. If the moved key can't be relocated because the other
/// slot of it is also occupied, another `relocation` is required and so on. If relocation is a very long chain
/// or meets a infinite loop, the table should be resized or rehashed.
///
pub struct LockFreeCuckooHash<K, V>
where
    K: Eq + Hash,
{
    /// The inner map will be replaced after resize.
    map: AtomicPtr<MapInner<K, V>>,
}

impl<K, V> std::fmt::Debug for LockFreeCuckooHash<K, V>
where
    K: std::fmt::Debug + Eq + Hash,
    V: std::fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = pin();
        self.load_inner(&guard).fmt(f)
    }
}

impl<K, V> Default for LockFreeCuckooHash<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Drop for LockFreeCuckooHash<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    fn drop(&mut self) {
        let guard = pin();
        self.load_inner(&guard).drop_entries(&guard);
        unsafe {
            self.map.load(Ordering::SeqCst, &guard).into_box();
        }
    }
}

impl<'guard, K, V> LockFreeCuckooHash<K, V>
where
    K: 'guard + Eq + Hash,
{
    /// The default capacity of a new `LockFreeCuckooHash` when created by `LockFreeHashMap::new()`.
    pub const DEFAULT_CAPACITY: usize = 16;

    /// Create an empty `LockFreeCuckooHash` with default capacity.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Creates an empty `LockFreeCuckooHash` with the specified capacity.
    #[must_use]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: AtomicPtr::new(MapInner::with_capacity(
                capacity,
                [RandomState::new(), RandomState::new()],
            )),
        }
    }

    /// Returns the capacity of this hash table.
    #[inline]
    pub fn capacity(&self) -> usize {
        let guard = pin();
        self.load_inner(&guard).capacity()
    }

    /// Returns the number of used slots of this hash table.
    #[inline]
    pub fn size(&self) -> usize {
        let guard = pin();
        self.load_inner(&guard).size()
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(10, 10);
    /// let guard = pin();
    /// let v = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v.unwrap(), 10);
    /// ```
    ///
    #[inline]
    pub fn search_with_guard(&self, key: &K, guard: &'guard Guard) -> Option<&'guard V> {
        self.load_inner(guard).search(key, guard)
    }

    /// Insert a new key-value pair into the hashtable. If the key has already been in the
    /// table, the value will be overridden.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(10, 10);
    /// let guard = pin();
    /// let v1 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v1.unwrap(), 10);
    /// map.insert(10, 20);
    /// let v2 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v2.unwrap(), 20);
    /// ```
    ///
    #[inline]
    pub fn insert(&self, key: K, value: V) {
        let guard = pin();
        self.insert_with_guard(key, value, &guard)
    }

    /// Insert a new key-value pair into the hashtable. If the key has already been in the
    /// table, the value will be overridden.
    /// Different from `insert(k, v)`, this method requires a user provided guard.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// map.insert_with_guard(10, 10, &guard);
    /// let v1 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v1.unwrap(), 10);
    /// map.insert_with_guard(10, 20, &guard);
    /// let v2 = map.search_with_guard(&10, &guard);
    /// assert_eq!(*v2.unwrap(), 20);
    /// ```
    ///
    #[inline]
    pub fn insert_with_guard(&self, key: K, value: V, guard: &'guard Guard) {
        let kvpair = SharedPtr::from_box(Box::new(KVPair { key, value }));
        loop {
            if !self.load_inner(guard).insert(kvpair, &self.map, guard) {
                // If `insert` returns false it means the hashmap has been
                // resized, we need to try to insert the kvpair again.
                continue;
            }
            return;
        }
    }

    /// Remove a key from the map.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// map.insert(10, 20);
    /// map.remove(&10);
    /// let value = map.search_with_guard(&10, &guard);
    /// assert_eq!(value.is_none(), true);
    /// ```
    ///
    #[inline]
    pub fn remove(&self, key: &K) -> bool {
        let guard = pin();
        self.remove_with_guard(key, &guard)
    }

    /// Remove a key from the map.
    /// Different from `remove(k)`, this method requires a user provided guard.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// map.insert(10, 20);
    /// map.remove_with_guard(&10, &guard);
    /// let value = map.search_with_guard(&10, &guard);
    /// assert_eq!(value.is_none(), true);
    /// ```
    ///
    #[inline]
    pub fn remove_with_guard(&self, key: &K, guard: &'guard Guard) -> bool {
        loop {
            let (succ, exists) = self.load_inner(guard).remove(key, &self.map, guard);
            if succ {
                return exists;
            }
        }
    }

    /// `load_inner` atomically loads the `MapInner` of hashmap.
    #[allow(clippy::unwrap_used)]
    fn load_inner(&self, guard: &'guard Guard) -> &'guard MapInner<K, V> {
        let raw = self.map.load(Ordering::SeqCst, guard).as_raw();
        // map is always not null, so the unsafe code is safe here.
        unsafe { raw.as_ref().unwrap() }
    }
}

#[cfg(test)]
#[allow(clippy::all, clippy::restriction)]
mod tests {
    use super::{pin, LockFreeCuckooHash};
    #[test]
    fn test_insert() {
        let hashtable = LockFreeCuckooHash::new();
        let key: u32 = 1;
        let value: u32 = 2;
        hashtable.insert(key, value);
        let guard = pin();
        let ret = hashtable.search_with_guard(&key, &guard);
        assert!(ret.is_some());
        assert_eq!(*(ret.unwrap()), value);
    }

    #[test]
    fn test_replace() {
        let hashtable = LockFreeCuckooHash::new();
        let key: u32 = 1;
        let value0: u32 = 2;
        hashtable.insert(key, value0);
        let guard = pin();
        let ret0 = hashtable.search_with_guard(&key, &guard);
        assert!(ret0.is_some());
        assert_eq!(*(ret0.unwrap()), value0);
        assert_eq!(hashtable.size(), 1);
        let value1: u32 = 3;
        hashtable.insert(key, value1);
        let ret1 = hashtable.search_with_guard(&key, &guard);
        assert!(ret1.is_some());
        assert_eq!(*(ret1.unwrap()), value1);
        assert_eq!(hashtable.size(), 1);
    }

    #[test]
    fn test_remove() {
        let hashtable = LockFreeCuckooHash::new();
        let key = 1;
        let value = 2;
        let fake_key = 3;
        hashtable.insert(key, value);
        assert_eq!(hashtable.size(), 1);
        assert!(!hashtable.remove(&fake_key));
        assert!(hashtable.remove(&key));
        assert_eq!(hashtable.size(), 0);
    }
}
