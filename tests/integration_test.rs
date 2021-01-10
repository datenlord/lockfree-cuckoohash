use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
use rand::Rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[test]
fn test_single_thread() {
    let capacity: usize = 100_000;
    let load_factor: f32 = 0.3;
    let remove_factor: f32 = 0.1;
    let size = (capacity as f32 * load_factor) as usize;

    let mut base_map: HashMap<u32, u32> = HashMap::with_capacity(capacity);
    let cuckoo_map: LockFreeCuckooHash<u32, u32> = LockFreeCuckooHash::with_capacity(capacity);

    let mut rng = rand::thread_rng();
    let guard = pin();

    for _ in 0..size {
        let key: u32 = rng.gen();
        let value: u32 = rng.gen();

        base_map.insert(key, value);
        cuckoo_map.insert_with_guard(key, value, &guard);

        let r: u8 = rng.gen();
        let need_remove = (r % 10) < ((remove_factor * 10_f32) as u8);
        if need_remove {
            base_map.remove(&key);
            cuckoo_map.remove_with_guard(&key, &guard);
        }
    }

    assert_eq!(base_map.len(), cuckoo_map.size());

    for (key, value) in base_map {
        let value2 = cuckoo_map.get(&key, &guard);
        assert_eq!(value, *value2.unwrap());
    }
}

#[test]
fn test_single_thread_resize() {
    let init_capacity: usize = 32;
    let size = 1024;

    let mut base_map: HashMap<u32, u32> = HashMap::with_capacity(init_capacity);
    let cuckoo_map: LockFreeCuckooHash<u32, u32> = LockFreeCuckooHash::with_capacity(init_capacity);

    let mut rng = rand::thread_rng();
    let guard = pin();

    for _ in 0..size {
        let mut key: u32 = rng.gen();
        while base_map.contains_key(&key) {
            key = rng.gen();
        }
        let value: u32 = rng.gen();

        base_map.insert(key, value);
        cuckoo_map.insert_with_guard(key, value, &guard);
    }

    assert_eq!(base_map.len(), cuckoo_map.size());
    assert_eq!(cuckoo_map.size(), size);
    for (key, value) in base_map {
        let value2 = cuckoo_map.get(&key, &guard);
        assert_eq!(value, *value2.unwrap());
    }
}

#[test]
fn test_multi_thread() {
    let capacity: usize = 1_000_000;
    let load_factor: f32 = 0.2;
    let num_thread: usize = 4;

    let size = (capacity as f32 * load_factor) as usize;
    let warmup_size = size / 3;

    let mut warmup_entries: Vec<(u32, u32)> = Vec::with_capacity(warmup_size);

    let mut new_insert_entries: Vec<(u32, u32)> = Vec::with_capacity(size - warmup_size);

    let mut base_map: HashMap<u32, u32> = HashMap::with_capacity(capacity);
    let cuckoo_map: LockFreeCuckooHash<u32, u32> = LockFreeCuckooHash::with_capacity(capacity);

    let mut rng = rand::thread_rng();
    let guard = pin();

    for _ in 0..warmup_size {
        let mut key: u32 = rng.gen();
        while base_map.contains_key(&key) {
            key = rng.gen();
        }
        let value: u32 = rng.gen();
        base_map.insert(key, value);
        cuckoo_map.insert_with_guard(key, value, &guard);
        warmup_entries.push((key, value));
    }

    for _ in 0..(size - warmup_size) {
        let mut key: u32 = rng.gen();
        while base_map.contains_key(&key) {
            key = rng.gen();
        }
        let value: u32 = rng.gen();
        new_insert_entries.push((key, value));
        base_map.insert(key, value);
    }

    let mut handles = Vec::with_capacity(num_thread);
    let insert_count = Arc::new(AtomicUsize::new(0));
    let cuckoo_map = Arc::new(cuckoo_map);
    let warmup_entries = Arc::new(warmup_entries);
    let new_insert_entries = Arc::new(new_insert_entries);
    for _ in 0..num_thread {
        let insert_count = insert_count.clone();
        let cuckoo_map = cuckoo_map.clone();
        let warmup_entries = warmup_entries.clone();
        let new_insert_entries = new_insert_entries.clone();
        let handle = std::thread::spawn(move || {
            let guard = pin();
            let mut entry_idx = insert_count.fetch_add(1, Ordering::SeqCst);
            let mut rng = rand::thread_rng();
            while entry_idx < new_insert_entries.len() {
                // read 5 pairs ,then insert 1 pair.
                for _ in 0..5 {
                    // let rnd_idx: usize = rng.gen_range(0, warmup_entries.len());
                    let rnd_idx: usize = rng.gen_range(0..warmup_entries.len());
                    let warmup_entry = &warmup_entries[rnd_idx];
                    let res = cuckoo_map.get(&warmup_entry.0, &guard);
                    assert_eq!(res.is_some(), true);
                    assert_eq!(*res.unwrap(), warmup_entry.1);
                }
                let insert_pair = &new_insert_entries[entry_idx];
                cuckoo_map.insert_with_guard(insert_pair.0, insert_pair.1, &guard);
                entry_idx = insert_count.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    for (k, v) in base_map {
        let v2 = cuckoo_map.get(&k, &guard);
        assert_eq!(v, *v2.unwrap());
    }
}

#[test]
fn test_multi_thread_resize() {
    let insert_per_thread = 100000;
    let num_thread = 8;
    let cuckoo_map: LockFreeCuckooHash<u32, u32> = LockFreeCuckooHash::with_capacity(8);
    let mut base_map: HashMap<u32, u32> = HashMap::new();
    let mut entries: Vec<(u32, u32)> = Vec::with_capacity(insert_per_thread * num_thread);

    let mut rng = rand::thread_rng();

    for _ in 0..(insert_per_thread * num_thread) {
        let mut key: u32 = rng.gen();
        while base_map.contains_key(&key) {
            key = rng.gen();
        }
        let value: u32 = rng.gen();
        base_map.insert(key, value);
        entries.push((key, value));
    }

    let mut handles = Vec::with_capacity(num_thread);
    let cuckoo_map = Arc::new(cuckoo_map);
    let entries = Arc::new(entries);
    for thread_idx in 0..num_thread {
        let cuckoo_map = cuckoo_map.clone();
        let entries = entries.clone();
        let handle = std::thread::spawn(move || {
            let guard = pin();
            let begin = thread_idx * insert_per_thread;
            for i in begin..(begin + insert_per_thread) {
                cuckoo_map.insert_with_guard(entries[i].0, entries[i].1, &guard);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let guard = pin();
    assert_eq!(base_map.len(), cuckoo_map.size());
    for (k, v) in base_map {
        let v2 = cuckoo_map.get(&k, &guard);
        assert!(v2.is_some());
        assert_eq!(v, *v2.unwrap());
    }
}
