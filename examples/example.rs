use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
use std::sync::Arc;

fn simple_read_write_example() {
    // Create a new empty hashmap.
    let hashtable = LockFreeCuckooHash::new();
    // or use LockFreeCuckooHash::with_capacity(capacity) to specify the max capacity.

    // Insert the key-value pair into the hashtable.
    let key = 1;
    let value = "value";
    hashtable.insert(key, value);

    // Search the value corresponding to the key.
    // `guard` is used to keep the current thread pinned.
    // If a `guard` is pinned, the returned value's reference
    // is always valid. In other words, other threads cannot destroy
    // `value` until all of the `guard`s are unpinned.
    let guard = pin();
    let result = hashtable.get(&key, &guard);
    assert!(result.is_some());
    assert_eq!(*result.unwrap(), value);

    let none_exist = hashtable.get(&2, &guard);
    assert!(none_exist.is_none());

    // Remove a key-value pair.
    let is_removed = hashtable.remove(&2);
    assert!(!is_removed);

    let is_removed = hashtable.remove(&key);
    assert!(is_removed);
}

fn multi_threads_read_write() {
    let hashtable = Arc::new(LockFreeCuckooHash::new());
    // Create 4 threads to write the hash table.
    let mut handles = Vec::with_capacity(4);
    for i in 0..4 {
        // Transfer the reference to each thread, no need for a mutex.
        let hashtable = hashtable.clone();
        let handle = std::thread::spawn(move || {
            for j in 0..100 {
                let key = i * 100 + j;
                let value = i;
                hashtable.insert(key, value);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let guard = pin();
    assert_eq!(hashtable.size(), 4 * 100);
    for i in 0..4 {
        for j in 0..100 {
            let key = i * 100 + j;
            let value = i;
            let ret = hashtable.get(&key, &guard);
            assert!(ret.is_some());
            assert_eq!(*ret.unwrap(), value);
        }
    }
}

fn main() {
    simple_read_write_example();

    multi_threads_read_write();
}
