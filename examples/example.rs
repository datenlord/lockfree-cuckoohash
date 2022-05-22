#![allow(clippy::bool_assert_comparison)] // FIXME

use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
use std::sync::Arc;

fn simple_read_write_example() {
    // Create a new empty map.
    let map = LockFreeCuckooHash::new();
    // or use LockFreeCuckooHash::with_capacity(capacity) to specify the max capacity.

    // `guard` is used to keep the current thread pinned.
    // If a `guard` is pinned, the returned value's reference
    // is always valid. In other words, other threads cannot destroy
    // `value` until all of the `guard`s are unpinned.
    let guard = pin();

    // Insert the key-value pair into the map.
    let key = 1;
    let value = "value";
    // The returned value indicates whether the map had the key before this insertion.
    assert_eq!(map.insert(key, value), false);
    assert_eq!(map.insert(key, "value2"), true);
    // If you want to get the replaced value, try `insert_with_guard`.
    assert_eq!(map.insert_with_guard(key, value, &guard), Some(&"value2"));

    // Search the value corresponding to the key.
    assert_eq!(map.get(&key, &guard), Some(&value));
    assert_eq!(map.get(&2, &guard), None);

    // Remove a key-value pair.
    // `remove` returns `false` if the map does not have the key.
    assert_eq!(map.remove(&2), false);
    assert_eq!(map.remove(&key), true);
    assert_eq!(map.remove(&key), false);

    // If you want to get the removed value, use `remove_with_guard` instead.
    map.insert(key, value);
    assert_eq!(map.remove_with_guard(&key, &guard), Some(&value));
    assert_eq!(map.remove_with_guard(&key, &guard), None);
}

fn multi_threads_read_write() {
    let map = Arc::new(LockFreeCuckooHash::new());
    // Create 4 threads to write the hash table.
    let mut handles = Vec::with_capacity(4);
    for i in 0..4 {
        // Transfer the reference to each thread, no need for a mutex.
        let map = map.clone();
        let handle = std::thread::spawn(move || {
            for j in 0..100 {
                let key = i * 100 + j;
                let value = i;
                map.insert(key, value);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let guard = pin();
    assert_eq!(map.size(), 4 * 100);
    for i in 0..4 {
        for j in 0..100 {
            let key = i * 100 + j;
            let value = i;
            let ret = map.get(&key, &guard);
            assert_eq!(ret, Some(&value));
        }
    }
}

fn main() {
    simple_read_write_example();

    multi_threads_read_write();
}
