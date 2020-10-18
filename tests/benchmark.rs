use lockfree_cuckoohash::{pin, LockFreeCuckooHash};
use rand::Rng;
use std::sync::Arc;
use std::time::Instant;

fn bench(
    init_capacity: usize,
    capacity: usize,
    warmup_factor: f32,
    write_factor: f32,
    remove_factor: f32,
    num_read_per_write: usize,
    num_thread: usize,
) -> (usize, f64) {
    assert!(warmup_factor < 1.0);
    assert!(write_factor + warmup_factor < 1.0);
    assert!(remove_factor < write_factor);

    let warmup_size = (capacity as f32 * warmup_factor) as usize;

    let cuckoo_map = Arc::new(LockFreeCuckooHash::with_capacity(init_capacity));

    let guard = pin();
    let mut rng = rand::thread_rng();

    for _ in 0..warmup_size {
        let key: u32 = rng.gen();
        let value: u32 = rng.gen();

        cuckoo_map.insert_with_guard(key, value, &guard);
    }

    let insert_per_thread = (capacity as f32 * write_factor) as usize / num_thread;
    let remove_per_thread = (capacity as f32 * remove_factor) as usize / num_thread;

    let mut insert_entries = Vec::with_capacity(insert_per_thread * num_thread);
    for _ in 0..(insert_per_thread * num_thread) {
        let key: u32 = rng.gen();
        let value: u32 = rng.gen();
        insert_entries.push((key, value));
    }
    let insert_entries = Arc::new(insert_entries);
    let mut handles = Vec::with_capacity(num_thread);
    let start = Instant::now();
    for i in 0..num_thread {
        let map = cuckoo_map.clone();
        let insert_entries = insert_entries.clone();
        let handle = std::thread::spawn(move || {
            let mut remove_flag: f32 = 0.0;
            let mut remove_idx = 0;
            let num_remove_per_thread = remove_factor / write_factor;
            let mut rng = rand::thread_rng();
            let guard = &pin();
            let insert_entries =
                &insert_entries[i * insert_per_thread..(i + 1) * insert_per_thread];
            for i in 0..insert_per_thread {
                // 1. insert a kv pair
                map.insert_with_guard(insert_entries[i].0, insert_entries[i].1, guard);

                // 2. read num_read_per_write kv pairs
                for _ in 0..num_read_per_write {
                    let key_idx: usize = rng.gen::<usize>() % (i + 1);
                    map.search_with_guard(&insert_entries[key_idx].0, guard);
                }

                // 3. remove num_remove_per_write kv pairs
                remove_flag += num_remove_per_thread;
                if remove_flag >= 1.0 {
                    map.remove_with_guard(&insert_entries[remove_idx].0, guard);
                    remove_flag -= 1.0;
                    remove_idx += 1;
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start.elapsed().as_millis() as f64;
    let ops = (insert_per_thread + remove_per_thread + insert_per_thread * num_read_per_write)
        * num_thread;
    guard.flush();
    (ops, duration)
}

fn bench_read_write(num_thread: usize) {
    let capacity = 10_000_000;
    let warmup_factor = 0.01;
    let insert_factor = 0.2;
    let remove_factor = 0.0;
    let num_read_per_write = 19;
    let (ops, duration) = bench(
        capacity,
        capacity,
        warmup_factor,
        insert_factor,
        remove_factor,
        num_read_per_write,
        num_thread,
    );

    let num_remove_per_write = remove_factor / insert_factor;

    println!(
        "num_thread: {}, init_capacity: {}, capacity: {}, ops: {}, search: {}%, insert: {}%, remove: {}%, duration: {}ms, throughput: {}op/ms",
        num_thread, capacity, capacity, ops,
        num_read_per_write as f32 / (num_read_per_write as f32 + num_remove_per_write + 1.0) * 100.0,
        1.0 / (num_read_per_write as f32 + num_remove_per_write + 1.0)* 100.0,
        num_remove_per_write as f32 / (num_read_per_write as f32 + num_remove_per_write + 1.0) * 100.0,
        duration, (ops as f64 / duration)
    );
}

#[test]
#[ignore]
fn bench_read_write_scale() {
    let cpu_cores = num_cpus::get();
    println!("Bench: search + insert, num_cpu_cores: {}", cpu_cores);
    let mut num_thread = 1;
    loop {
        if num_thread > cpu_cores {
            break;
        }
        bench_read_write(num_thread);
        num_thread *= 2;
    }
}

fn bench_insert_only(num_thread: usize) {
    let capacity = 10_000_000;
    let warmup_factor = 0.0;
    let insert_factor = 0.4;
    let remove_factor = 0.0;
    let num_read_per_write = 0;
    let (ops, duration) = bench(
        capacity,
        capacity,
        warmup_factor,
        insert_factor,
        remove_factor,
        num_read_per_write,
        num_thread,
    );
    let guard = pin();
    guard.flush();
    println!(
        "num_thread: {}, init_capacity: {}, capacity: {}, insert: {}, duration: {}ms, throughput: {}op/ms",
        num_thread, capacity, capacity, insert_factor * capacity as f32,
        duration, (ops as f64 / duration)
    );
}

#[test]
#[ignore]
fn bench_insert_scale() {
    let cpu_cores = num_cpus::get();
    println!("Bench: insert only, num_cpu_cores: {}", cpu_cores);
    let mut num_thread = 1;
    loop {
        if num_thread > cpu_cores {
            break;
        }
        bench_insert_only(num_thread);
        num_thread *= 2;
    }
}

fn bench_insert_resize(num_thread: usize) {
    let init_capacity = 5_000_000;
    let capacity = 10_000_000;
    let warmup_factor = 0.0;
    let insert_factor = 0.5;
    let remove_factor = 0.0;
    let num_read_per_write = 0;
    let (ops, duration) = bench(
        init_capacity,
        capacity,
        warmup_factor,
        insert_factor,
        remove_factor,
        num_read_per_write,
        num_thread,
    );
    let guard = pin();
    guard.flush();
    println!(
        "num_thread: {}, init_capacity: {}, capacity: {}, insert: {}, duration: {}ms, throughput: {}op/ms",
        num_thread, init_capacity, capacity, insert_factor * capacity as f32,
        duration, (ops as f64 / duration)
    );
}

#[test]
#[ignore]
fn bench_resize_scale() {
    let cpu_cores = num_cpus::get();
    println!("Bench: insert with resize, num_cpu_cores: {}", cpu_cores);
    let mut num_thread = 1;
    loop {
        if num_thread > cpu_cores {
            break;
        }
        bench_insert_resize(num_thread);
        num_thread *= 2;
    }
}
