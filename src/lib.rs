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

use pointer::{AtomicPtr, SharedPtr};
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::Hash;
use std::sync::atomic::Ordering;

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
    map: AtomicPtr<map_inner::MapInner<K, V>>,
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
            map: AtomicPtr::new(map_inner::MapInner::with_capacity(
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

    /// # Safety
    ///
    /// Clear the hashmap with the specified capacity.
    /// The caller must make sure the hashmap is not during a resize.
    #[inline]
    pub unsafe fn clear(&self) {
        let cap = self.capacity();
        self.clear_with_capacity(cap);
    }

    /// # Safety
    ///
    /// Clear the hashmap with the specified capacity.
    /// The caller must make sure the hashmap is not during a resize.
    #[inline]
    pub unsafe fn clear_with_capacity(&self, capacity: usize) {
        let guard = pin();
        let new_map = SharedPtr::from_box(Box::new(map_inner::MapInner::<K, V>::with_capacity(
            capacity,
            [RandomState::new(), RandomState::new()],
        )));
        loop {
            let current_map = self.map.load(Ordering::SeqCst, &guard);
            match self
                .map
                .compare_and_set(current_map, new_map, Ordering::SeqCst, &guard)
            {
                Ok(old_map) => {
                    guard.defer_unchecked(move || {
                        old_map.into_box();
                    });
                    break;
                }
                Err(_) => {
                    continue;
                }
            }
        }
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(1, "a");
    /// let guard = pin();
    /// let v = map.get(&1, &guard);
    /// assert_eq!(v, Some(&"a"));
    /// ```
    ///
    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q, guard: &'guard Guard) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.load_inner(guard)
            .search(key, guard)
            .map(|pair| &pair.value)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// # Example
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(1, "a");
    /// let guard = pin();
    /// let v = map.get_key_value(&1, &guard);
    /// assert_eq!(v, Some((&1, &"a")));
    /// ```
    ///
    #[inline]
    pub fn get_key_value<Q: ?Sized>(
        &self,
        key: &Q,
        guard: &'guard Guard,
    ) -> Option<(&'guard K, &'guard V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.load_inner(guard)
            .search(key, guard)
            .map(|pair| (&pair.key, &pair.value))
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// # Example
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    ///
    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let guard = pin();
        self.get_key_value(key, &guard).is_some()
    }

    /// Insert a new key-value pair into the map.
    /// If the map did not have this key present, `false` is returned.
    /// If the map did have this key present, the value is updated, and `true` is returned.
    /// If you want to get the replaced value, try `insert_with_guard` instead.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// assert_eq!(map.insert(1, "a"), false);
    /// assert_eq!(map.insert(2, "b"), false);
    /// assert_eq!(map.insert(1, "aaa"), true);
    /// ```
    ///
    #[inline]
    pub fn insert(&self, key: K, value: V) -> bool {
        let guard = pin();
        self.insert_with_guard(key, value, &guard).is_some()
    }

    /// Insert a new key-value pair into the map.
    /// If the map did not have this key present, `None` is returned.
    /// If the map did have this key present, the value is updated, and the reference to the old value is returned.
    /// Different from `insert(k, v)`, this method requires a user provided guard.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// assert_eq!(map.insert_with_guard(1, "a", &guard), None);
    /// assert_eq!(map.insert_with_guard(2, "b", &guard), None);
    /// assert_eq!(map.insert_with_guard(1, "abc", &guard), Some(&"a"));
    /// ```
    ///
    #[inline]
    pub fn insert_with_guard(&self, key: K, value: V, guard: &'guard Guard) -> Option<&'guard V> {
        let kvpair = SharedPtr::from_box(Box::new(map_inner::KVPair { key, value }));
        loop {
            match self.load_inner(guard).insert(
                kvpair,
                map_inner::InsertType::InsertOrReplace,
                &self.map,
                guard,
            ) {
                // If `insert` returns false it means the hashmap has been
                // resized, we need to try to insert the kvpair again.
                map_inner::WriteResult::Retry => continue,
                map_inner::WriteResult::Succ(result) => return result.map(|pair| &pair.value),
            }
        }
    }

    /// Insert a new key-value pair into the map if the map does not contain the key.
    /// If the map contains the key, return the old value.
    /// If the map does not contain the key, insert the new key-value, and return the new value.
    ///
    /// Notice: When two concurrent `get_or_insert` methods are trying to insert the same key,
    /// only one will succeed. But if a `get_or_insert` and a `insert` are called simultaneously with
    /// the same key, the `get_or_insert` may still can insert the key-value pair even if `insert` has
    /// already succeeded.
    ///
    /// An example for concurrent `get_or_insert`s：
    ///
    ///#         Thread A              |        Thread B
    ///#  call `get_or_insert(key, A)` | call `get_or_insert(key, B)`
    ///#                               |
    ///#     return value = A          |      
    ///#                               |     return value = A
    ///
    /// We can see, only one thread can insert the key-value, and the other will return the old value.
    ///
    /// An example for concurrent `get_or_insert` and `insert`:
    ///
    ///#         Thread A              |        Thread B
    ///#  call `get_or_insert(key, A)` |   call `insert(key, B)`
    ///#                               |      return value = B
    ///#     return value = A          |       
    ///#                               |   call `get(key, A)`
    ///#                               |      return value = A
    ///
    /// We can see here, even if Thread B has already inserted (key, B) into the map, but Thread A can
    /// still insert (key, A), which is not consistent with the semantics of `get_or_insert`.      
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// assert_eq!(map.get_or_insert(1, "a", &guard), &"a");
    /// assert_eq!(map.get_or_insert(1, "b", &guard), &"a");
    ///
    /// ```
    #[inline]
    #[allow(clippy::unwrap_used)]
    pub fn get_or_insert(&self, key: K, value: V, guard: &'guard Guard) -> &'guard V {
        let kvpair = SharedPtr::from_box(Box::new(map_inner::KVPair { key, value }));
        loop {
            match self.load_inner(guard).insert(
                kvpair,
                map_inner::InsertType::GetOrInsert,
                &self.map,
                guard,
            ) {
                // If `insert` returns false it means the hashmap has been
                // resized, we need to try to insert the kvpair again.
                map_inner::WriteResult::Retry => continue,
                map_inner::WriteResult::Succ(result) => {
                    return &result
                        .unwrap_or(unsafe { kvpair.as_raw().as_ref().unwrap() })
                        .value;
                }
            }
        }
    }

    /// Insert a new key-value pair into the map if the map does not contain the key.
    ///
    /// Notice: similar to `get_or_insert`, when two concurent `insert_if_not_exists` are
    /// called, only one will succeed. But when concurrent `insert_if_not_exists` and `insert`
    /// are called, `insert_if_not_exists` may still succeed even if `insert` has already inserted
    /// the pair.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// assert_eq!(map.insert_if_not_exists(1, "a"), true);
    /// assert_eq!(map.get(&1, &guard), Some(&"a"));
    /// assert_eq!(map.insert_if_not_exists(1, "b"), false);
    /// assert_eq!(map.get(&1, &guard), Some(&"a"));
    /// ```
    #[inline]
    pub fn insert_if_not_exists(&self, key: K, value: V) -> bool {
        let guard = &pin();
        let kvpair = SharedPtr::from_box(Box::new(map_inner::KVPair { key, value }));
        loop {
            match self.load_inner(guard).insert(
                kvpair,
                map_inner::InsertType::GetOrInsert,
                &self.map,
                guard,
            ) {
                // If `insert` returns false it means the hashmap has been
                // resized, we need to try to insert the kvpair again.
                map_inner::WriteResult::Retry => continue,
                map_inner::WriteResult::Succ(result) => return result.is_none(),
            }
        }
    }

    /// Compare the cuurent value with `old_value`, update the value to `new_value` if
    /// they are equal.
    /// This method returns true if the update succeeds, otherwise returns false.
    ///     
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// let guard = pin();
    /// assert_eq!(map.insert(1, "a"), false);
    /// assert_eq!(map.compare_and_update(1, "c", &"b"), false);
    /// assert_eq!(map.get(&1, &guard), Some(&"a"));
    /// assert_eq!(map.compare_and_update(1, "c", &"a"), true);
    /// assert_eq!(map.get(&1, &guard), Some(&"c"));
    #[inline]
    pub fn compare_and_update(&self, key: K, new_value: V, old_value: &V) -> bool
    where
        V: PartialEq,
    {
        let guard = &pin();
        let kvpair = SharedPtr::from_box(Box::new(map_inner::KVPair {
            key,
            value: new_value,
        }));
        let update_func: fn(&V, &V) -> bool = V::eq;
        loop {
            match self.load_inner(guard).insert(
                kvpair,
                map_inner::InsertType::CompareAndUpdate(old_value, update_func),
                &self.map,
                guard,
            ) {
                // If `insert` returns false it means the hashmap has been
                // resized, we need to try to insert the kvpair again.
                map_inner::WriteResult::Retry => continue,
                map_inner::WriteResult::Succ(res) => return res.is_some(),
            }
        }
    }

    /// Removes a key from the map, returning `true` if the key was previously in the map.
    /// If you want to get the old value, try `map.remove_with_guard()` instead.
    ///
    /// # Example:
    ///
    /// ```
    /// use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
    /// let map = LockFreeCuckooHash::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&2), false);
    /// assert_eq!(map.remove(&1), true);
    /// assert_eq!(map.remove(&1), false);
    /// ```
    ///
    #[inline]
    pub fn remove<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let guard = pin();
        self.remove_with_guard(key, &guard).is_some()
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
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_with_guard(&2, &guard), None);
    /// assert_eq!(map.remove_with_guard(&1, &guard), Some(&"a"));
    /// assert_eq!(map.remove_with_guard(&1, &guard), None);
    /// ```
    ///
    #[inline]
    pub fn remove_with_guard<Q: ?Sized>(&self, key: &Q, guard: &'guard Guard) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        loop {
            match self.load_inner(guard).remove(key, &self.map, guard) {
                map_inner::WriteResult::Retry => continue,
                map_inner::WriteResult::Succ(old) => return old.map(|pair| &pair.value),
            }
        }
    }

    /// `load_inner` atomically loads the `MapInner` of hashmap.
    #[allow(clippy::unwrap_used)]
    fn load_inner(&self, guard: &'guard Guard) -> &'guard map_inner::MapInner<K, V> {
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
        let ret = hashtable.get(&key, &guard);
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
        let ret0 = hashtable.get(&key, &guard);
        assert!(ret0.is_some());
        assert_eq!(*(ret0.unwrap()), value0);
        assert_eq!(hashtable.size(), 1);
        let value1: u32 = 3;
        hashtable.insert(key, value1);
        let ret1 = hashtable.get(&key, &guard);
        assert!(ret1.is_some());
        assert_eq!(*(ret1.unwrap()), value1);
        assert_eq!(hashtable.size(), 1);
    }

    #[test]
    fn test_get_or_insert() {
        let hashtable = LockFreeCuckooHash::new();
        let guard = &pin();
        hashtable.insert(1, 1);
        assert_eq!(hashtable.get_or_insert(1, 2, guard), &1);
        assert_eq!(hashtable.get_or_insert(2, 3, guard), &3);
        assert_eq!(hashtable.get_or_insert(2, 4, guard), &3);
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

    #[test]
    fn test_clear() {
        let hashtable = LockFreeCuckooHash::new();
        let key = 1;
        let value = 2;
        hashtable.insert(key, value);
        let guard = pin();
        let ret = hashtable.get(&key, &guard);
        assert!(ret.is_some());
        assert_eq!(*(ret.unwrap()), value);
        unsafe { hashtable.clear() };
        let ret = hashtable.get(&key, &guard);
        assert!(ret.is_none());
    }

    #[test]
    fn test_compare_and_update() {
        let hashtable = LockFreeCuckooHash::new();
        let guard = &pin();
        hashtable.insert(1, 10);
        assert_eq!(hashtable.compare_and_update(1, 20, &30), false);
        assert_eq!(hashtable.get(&1, guard), Some(&10));
        assert_eq!(hashtable.compare_and_update(1, 20, &10), true);
        assert_eq!(hashtable.get(&1, guard), Some(&20));
    }
}
