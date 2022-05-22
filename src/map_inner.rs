#![allow(clippy::indexing_slicing)] // TODO: use safe method for indexing and remove this line.

use super::pointer::{AtomicPtr, SharedPtr};
use clippy_utilities::{Cast, OverflowArithmetic};
use crossbeam_epoch::{pin, Guard};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{borrow::Borrow, collections::hash_map::RandomState};

/// `KVPair` contains the key-value pair.
#[derive(Debug)]
pub struct KVPair<K, V> {
    // TODO: maybe cache both hash keys here.
    /// `key` is the key of a KV pair.
    pub key: K,
    /// `value` is the value of a KV pair.
    pub value: V,
}

/// `SlotIndex` represents the index of a slot inside the hashtable.
/// The slot index is composed by `tbl_idx` and `slot_idx`.
#[derive(Clone, Copy, Debug)]
struct SlotIndex {
    /// `tbl_idx` is the index of the table.
    tbl_idx: usize,
    /// `slot_idx` is the index of the slot inside one table.
    slot_idx: usize,
}

/// `SlotState` represents the state of a slot.
/// A slot could be in one of the four states: null, key, reloc and copied.
#[derive(PartialEq)]
enum SlotState {
    /// `NullOrKey` means a slot is empty(null) or is occupied by a key-value
    /// pair normally without any other flags.
    NullOrKey = 0,
    /// `Reloc` means a slot is being relocated to the other slot.
    Reloc = 1,
    /// `Copied` means a slot is being copied to the new map during resize or
    /// has been copied to the new map.
    Copied = 2,
}

impl SlotState {
    /// `into_u8` converts a `SlotState` to `u8`.
    #[allow(clippy::as_conversions)]
    const fn into_u8(self) -> u8 {
        self as u8
    }

    /// `from_u8` converts a `u8` to `SlotState`.
    fn from_u8(state: u8) -> Self {
        match state {
            0 => Self::NullOrKey,
            1 => Self::Reloc,
            2 => Self::Copied,
            _ => panic!("Invalid slot state from u8: {}", state),
        }
    }
}

/// `InsertResult` is the returned type of the method `MapInner.insert()`
pub enum InsertResult<'guard, K, V> {
    /// The insert operation succeeded, returns `Some(&KVPair)` if the map had the key,
    /// otherwise returns `None` if the map does not have that key.
    Succ(Option<&'guard KVPair<K, V>>),
    /// The insert operation failed, returns `Some(&KVPair)` if the map had the key,
    /// otherwise returns `None` if the map does not have that key.
    Fail(Option<&'guard KVPair<K, V>>),
    /// The insert operation might fail because the hashmap has been resized by other writers,
    /// so the caller need to retry the insert again.
    Retry,
}

/// `RemoveResult` is the returned type of the method `MapInner.remove()`.
pub enum RemoveResult<'guard, K, V> {
    /// The remove operation succeeds, returns `Some(&KVPair)` if the map had the key,
    /// otherwise returns `None` if the map does not have that key.
    Succ(Option<&'guard KVPair<K, V>>),
    /// The remove operation might fail because the hashmap has been resized by other writers,
    /// so the caller need to retry the insert again.
    Retry,
}

/// `InsertType` is the type of a insert operation.
pub enum InsertType<'guard, V> {
    /// Insert a new key-value pair if the key does not exist, otherwise replace it.
    InsertOrReplace,
    /// Get the key-value pair if the key exists, otherwise insert a new one.
    GetOrInsert,
    /// Compare the current value with the expected one, update it to the new value
    /// if they are equal.
    /// The parameters for this item are the old_value and compare function
    CompareAndUpdate(&'guard V, fn(&V, &V) -> bool),
    /// Check the current value with the new value, update it to the new value
    ///  if the check function return true
    /// The parameters for this item is
    /// 1. the check function
    /// 2. whether need to insert if the key doesn't exist
    UpdateOn(fn(&V, &V) -> bool, bool),
}

/// `FindResult` is the returned type of the method `MapInner.find()`.
/// For more details, see `MapInner.find()`.
struct FindResult<'guard, K, V> {
    /// the table index of the slot that has the same key
    tbl_idx: Option<usize>,
    /// the first slot
    slot0: SharedPtr<'guard, KVPair<K, V>>,
    /// the second slot
    slot1: SharedPtr<'guard, KVPair<K, V>>,
}

/// `RelocateResult` is the returned type of the method `MapInner.relocate()`.
enum RelocateResult {
    /// The relocation succeeds.
    Succ,
    /// The relocation fails because the cuckoo path is too long or
    /// meets a dead loop. A resize is required for the new insertion.
    NeedResize,
    /// The map has been resized, should try to insert to the new map.
    Resized,
}

/// `MapInner` is the inner implementation of the `LockFreeCuckooHash`.
pub struct MapInner<K, V> {
    // TODO: support customized hasher.
    /// `hash_builders` is used to hash the keys.
    hash_builders: [RandomState; 2],
    /// `tables` contains the key-value pairs.
    tables: Vec<Vec<AtomicPtr<KVPair<K, V>>>>,
    /// `size` is the number of inserted pairs of the hash map.
    size: AtomicUsize,

    // For resize
    /// `next_copy_idx` is the next slot idx which need to be copied.
    next_copy_idx: AtomicUsize,
    /// `copied_num` is the number of copied slots.
    copied_num: AtomicUsize,
    /// `new_map` is the resized new map.
    new_map: AtomicPtr<MapInner<K, V>>,
}

impl<K, V> std::fmt::Debug for MapInner<K, V>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    // This is not thread-safe.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let capacity = self.tables[0].len();
        let guard = pin();
        let mut f = f.debug_map();
        for tbl_idx in 0..2 {
            for slot_idx in 0..capacity {
                let slot = self.tables[tbl_idx][slot_idx].load(Ordering::SeqCst, &guard);
                unsafe {
                    if let Some(kv) = slot.as_raw().as_ref() {
                        f.entry(&kv.key, &kv.value);
                    }
                }
            }
        }
        f.finish()
    }
}

impl<'guard, K, V> MapInner<K, V>
where
    K: 'guard + Eq + Hash,
{
    /// `with_capacity` creates a new `MapInner` with specified capacity.
    pub fn with_capacity(capacity: usize, hash_builders: [RandomState; 2]) -> Self {
        let single_table_capacity = match capacity.checked_add(1) {
            Some(capacity) => capacity.overflow_div(2),
            None => capacity.overflow_div(2),
        };
        let mut tables = Vec::with_capacity(2);

        for _ in 0_u32..2 {
            let mut table = Vec::with_capacity(single_table_capacity);
            for _ in 0..single_table_capacity {
                table.push(AtomicPtr::null());
            }
            tables.push(table);
        }

        Self {
            hash_builders,
            tables,
            size: AtomicUsize::new(0),
            next_copy_idx: AtomicUsize::new(0),
            copied_num: AtomicUsize::new(0),
            new_map: AtomicPtr::null(),
        }
    }

    /// `capacity` returns the current capacity of the hash map.
    pub fn capacity(&self) -> usize {
        self.tables[0].len().overflow_mul(2)
    }

    /// `size` returns the number of inserted pairs of the hash map.
    pub fn size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    /// `search` searches the value corresponding to the key.
    pub fn search<Q: ?Sized>(&self, key: &Q, guard: &'guard Guard) -> Option<&'guard KVPair<K, V>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        // TODO: the second hash value could be lazily evaluated.
        let slot_idx = vec![self.get_index(0, key), self.get_index(1, key)];
        // Because other concurrent `insert` operations may relocate the key during
        // our `search` here, we may miss the key with one-round query.
        // For example, suppose the key is located in `table[1][hash1(key)]` at first:
        //
        //      search thread              |    relocate thread
        //                                 |
        //   e1 = table[0][hash0(key)]     |
        //                                 | relocate key from table[1] to table[0]
        //   e2 = table[1][hash1(key)]     |
        //                                 |
        //   both e1 and e2 are empty      |
        // -> key not exists, return None  |

        // So `search` uses a two-round query to deal with the `missing key` problem.
        // But it is not enough because a relocation operation might interleave in between.
        // The other technique to deal with it is a logic-clock based counter -- `relocation count`.
        // Each slot contains a counter that records the number of relocations at the slot.
        loop {
            let mut counters = Vec::with_capacity(4);
            for i in 0_usize..4 {
                let (counter, entry, _) = self.get_entry(slot_idx[i.overflowing_rem(2).0], guard);
                if let Some(pair) = entry {
                    if key.eq(pair.key.borrow()) {
                        return entry;
                    }
                }
                counters.push(counter);
            }
            // Check the counter.
            if Self::check_counter(counters[0], counters[1], counters[2], counters[3]) {
                continue;
            }
            break;
        }
        None
    }

    /// Insert a new key-value pair into the hashtable. If the key has already been in the
    /// table, the value will be overridden.
    /// If the insert operation fails because of the map has been resized, this method returns
    /// `WriteResult::Retry`, and the caller need to retry.
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    pub fn insert(
        &self,
        kvpair: SharedPtr<'guard, KVPair<K, V>>,
        insert_type: InsertType<'guard, V>,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) -> InsertResult<'guard, K, V> {
        let mut new_slot = kvpair;
        let (_, new_entry, _) = Self::unwrap_slot(new_slot);
        // new_entry is just created from `key`, so the unwrap() is safe here.
        let (new_key, new_value) = if let Some(pair) = new_entry {
            (&pair.key, &pair.value)
        } else {
            return InsertResult::Retry;
        };
        let slot_idx0 = self.get_index(0, new_key);
        let slot_idx1 = self.get_index(1, new_key);
        loop {
            let find_result = self.find(new_key, slot_idx0, slot_idx1, outer_map, guard);
            let (tbl_idx, slot0, slot1) = match find_result {
                Some(r) => (r.tbl_idx, r.slot0, r.slot1),
                None => return InsertResult::Retry,
            };
            let (slot_idx, target_slot, key_exist) = match tbl_idx {
                Some(tbl_idx) => {
                    // The key has already been in the table, we need to replace the value.
                    if tbl_idx == 0 {
                        (Some(&slot_idx0), slot0, true)
                    } else {
                        (Some(&slot_idx1), slot1, true)
                    }
                }
                None => {
                    // The key is a new one, check if we have an empty slot.
                    if Self::slot_is_empty(slot0) {
                        (Some(&slot_idx0), slot0, false)
                    } else if Self::slot_is_empty(slot1) {
                        (Some(&slot_idx1), slot1, false)
                    } else {
                        // Both slots are occupied, we need a relocation.
                        (None, slot0, false)
                    }
                }
            };

            let mut need_relocate = false;

            if let Some(slot_idx) = slot_idx {
                // We found the key exists or we have an empty slot,
                // just replace the slot with the new one.

                match insert_type {
                    InsertType::GetOrInsert => {
                        if key_exist {
                            // The insert type is `GetOrInsert`, but the key exists.
                            // So we return with a `Get` semantic.
                            // The new inserted key-value could be dropped immediately
                            // since no one can read it.
                            // SAFETY: new_slot is guaranteed to be non-null.
                            unsafe {
                                drop(new_slot.into_box());
                            }

                            return InsertResult::Fail(Self::unwrap_slot(target_slot).1);
                        }
                        if slot_idx.tbl_idx != 0 {
                            // GetOrInsert only inserts key-value pair into the primary slot.
                            // So if the primary slot is not empty, we force trigger a relocation.
                            need_relocate = true;
                        }
                    }
                    InsertType::CompareAndUpdate(old_value, compare_fn) => {
                        // The insert type is `compareAndUpdate`, so we need to check if
                        // the current value equals to the old_v.
                        if !key_exist {
                            // SAFETY: new_slot is guaranteed to be non-null.
                            unsafe {
                                drop(new_slot.into_box());
                            }
                            return InsertResult::Fail(None);
                        }
                        let (_, entry, _) = Self::unwrap_slot(target_slot);
                        if let Some(current_pair) = entry {
                            if !compare_fn(old_value, &current_pair.value) {
                                // SAFETY: new_slot is guaranteed to be non-null.
                                unsafe {
                                    drop(new_slot.into_box());
                                }
                                return InsertResult::Fail(Some(current_pair));
                            }
                        }
                    }
                    InsertType::UpdateOn(compare_fn, force_insert) => {
                        if !key_exist && !force_insert {
                            // SAFETY: new_slot is guaranteed to be non-null.
                            unsafe {
                                drop(new_slot.into_box());
                            }
                            return InsertResult::Fail(None);
                        }
                        let (_, entry, _) = Self::unwrap_slot(target_slot);
                        if let Some(current_pair) = entry {
                            if !compare_fn(&current_pair.value, new_value) {
                                // SAFETY: new_slot is guaranteed to be non-null.
                                unsafe {
                                    drop(new_slot.into_box());
                                }
                                return InsertResult::Fail(Some(current_pair));
                            }
                        }
                    }
                    InsertType::InsertOrReplace => {}
                }

                if !need_relocate {
                    // update the relocation count.
                    new_slot = Self::set_rlcount(new_slot, Self::get_rlcount(target_slot), guard);

                    match self.tables[slot_idx.tbl_idx][slot_idx.slot_idx].compare_and_set(
                        target_slot,
                        new_slot,
                        Ordering::SeqCst,
                        guard,
                    ) {
                        Ok(old_slot) => {
                            if !key_exist {
                                self.size.fetch_add(1, Ordering::SeqCst);
                                return InsertResult::Succ(None);
                            }
                            if old_slot.as_raw() != new_slot.as_raw() {
                                Self::defer_drop_ifneed(old_slot, guard);
                            }
                            return InsertResult::Succ(Self::unwrap_slot(old_slot).1);
                        }
                        Err(err) => {
                            new_slot = err.1; // the snapshot is not valid, try again.
                            continue;
                        }
                    }
                }
            } else {
                need_relocate = true;
            }

            if need_relocate {
                // We meet a hash collision here, relocate the first slot.
                match self.relocate(slot_idx0, outer_map, guard) {
                    RelocateResult::Succ => continue,
                    RelocateResult::NeedResize => {
                        self.resize(outer_map, guard);
                        return InsertResult::Retry;
                    }
                    RelocateResult::Resized => {
                        return InsertResult::Retry;
                    }
                }
            }
        }
    }

    /// Remove a key from the map.
    /// If the remove operation fails because of the map has been resized, this method returns
    /// `RemoveResult::Retry`, and the caller need to retry. Otherwise, it will return the removed
    /// value if the key exists, or `None` if not.
    pub fn remove<Q: ?Sized>(
        &self,
        key: &Q,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) -> RemoveResult<'guard, K, V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        // TODO: we can return the removed value.
        let slot_idx0 = self.get_index(0, key);
        let slot_idx1 = self.get_index(1, key);
        let new_slot = SharedPtr::null();
        loop {
            let find_result = self.find(key, slot_idx0, slot_idx1, outer_map, guard);
            let (tbl_idx, slot0, slot1) = match find_result {
                Some(r) => (r.tbl_idx, r.slot0, r.slot1),
                None => return RemoveResult::Retry,
            };
            let tbl_idx = match tbl_idx {
                Some(idx) => idx,
                None => return RemoveResult::Succ(None), // The key does not exist.
            };
            if tbl_idx == 0 {
                Self::set_rlcount(new_slot, Self::get_rlcount(slot0), guard);
                match self.tables[0][slot_idx0.slot_idx].compare_and_set(
                    slot0,
                    new_slot,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Ok(old_slot) => {
                        self.size.fetch_sub(1, Ordering::SeqCst);
                        Self::defer_drop_ifneed(old_slot, guard);
                        return RemoveResult::Succ(Self::unwrap_slot(old_slot).1);
                    }
                    Err(_) => continue,
                }
            } else {
                if self.tables[0][slot_idx0.slot_idx]
                    .load(Ordering::SeqCst, guard)
                    .as_raw()
                    != slot0.as_raw()
                {
                    continue;
                }
                Self::set_rlcount(new_slot, Self::get_rlcount(slot1), guard);
                match self.tables[1][slot_idx1.slot_idx].compare_and_set(
                    slot1,
                    new_slot,
                    Ordering::SeqCst,
                    guard,
                ) {
                    Ok(old_slot) => {
                        self.size.fetch_sub(1, Ordering::SeqCst);
                        Self::defer_drop_ifneed(old_slot, guard);
                        return RemoveResult::Succ(Self::unwrap_slot(old_slot).1);
                    }
                    Err(_) => continue,
                }
            }
        }
    }

    /// `find` is similar to `search`, which searches the value corresponding to the key.
    /// The differences are:
    /// 1. `find` will help the relocation if the slot is marked.
    /// 2. `find` will dedup the duplicated keys.
    /// 3. `find` returns an option of `FindResult`. If it return `None`, it means the hashmap
    /// has been resized, and the caller need to retry the operation. Otherwise the `FindResult`
    /// contains three values:
    ///     a> the table index of the slot that has the same key.
    ///     b> the first slot.
    ///     c> the second slot.
    fn find<Q: ?Sized>(
        &self,
        key: &Q,
        slot_idx0: SlotIndex,
        slot_idx1: SlotIndex,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) -> Option<FindResult<'guard, K, V>>
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        loop {
            // Similar to `search`, `find` also uses a two-round search protocol.
            // If either of the two rounds finds a slot that contains the key, this method
            // returns the table index of that slot.
            // If both of the two rounds cannot find the key, we will check the relocation
            // count to decide whether we need to retry the two-round search.

            // If `try_find` meets a copied slot (during resize), it will help the resize and
            // then returns the `resize0` as true.
            // If `try_find` meets a slot which is being relocated, it will help the relocation
            // and then returns the `reloc0` as true. Notice that, if the relocation fails because
            // `help_relocate` meets a copied slot, `try_find` will return `resize0` as true.
            let (fr0, reloc0, resize0) = self.try_find(key, slot_idx0, slot_idx1, outer_map, guard);

            // `try_find` helps finish a resize, so return `None` to let the caller retry
            // the opertion with the new resized map.
            if resize0 {
                return None;
            }

            // `try_find` successfully helps a relocation, so we continue the loop to retry the
            // two-round search.
            if reloc0 {
                continue;
            }

            // The first round successfully finds a slot that contains the key, so we don't need
            // the second round now, just return the result here.
            if fr0.tbl_idx.is_some() {
                return Some(fr0);
            }

            // Otherwise, we need try the second round.
            let (fr1, reloc1, resize1) = self.try_find(key, slot_idx0, slot_idx1, outer_map, guard);
            if resize1 {
                return None;
            }
            if reloc1 {
                continue;
            }
            if fr1.tbl_idx.is_some() {
                return Some(fr1);
            }

            // Neither of the tow rounds can find the key, we check the relocation count to determine
            // whether we need to retry.
            if !Self::check_counter(
                Self::get_rlcount(fr0.slot0),
                Self::get_rlcount(fr0.slot1),
                Self::get_rlcount(fr1.slot0),
                Self::get_rlcount(fr1.slot1),
            ) {
                return Some(FindResult {
                    tbl_idx: None,
                    slot0: fr1.slot0,
                    slot1: fr1.slot1,
                });
            }
        }
    }

    /// `try_find` tries to find the key in the hash map (only once).
    /// The returned values are:
    /// 1. The find result, including the index of the slot which contains the key, and both slots of the key.
    /// 2. Whether we successfully help a relocation.
    /// 3. Whether the map has been resized.
    fn try_find<Q: ?Sized>(
        &self,
        key: &Q,
        slot_idx0: SlotIndex,
        slot_idx1: SlotIndex,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) -> (FindResult<'guard, K, V>, bool, bool)
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        let mut result = FindResult {
            tbl_idx: None,
            slot0: SharedPtr::null(),
            slot1: SharedPtr::null(),
        };
        // Read the first slot at first.
        let slot0 = self.get_slot(slot_idx0, guard);
        let (_, entry0, state0) = Self::unwrap_slot(slot0);
        match state0 {
            SlotState::Reloc => {
                // The slot is being relocated, we help this relocation.
                if self.help_relocate(slot_idx0, false, true, outer_map, guard) {
                    // The relocation succeeds, we set the second returned value as `true`.
                    return (result, true, false);
                } else {
                    // The relocation fails, because the table has been resized.
                    // We set the third returned value as `true`.
                    return (result, false, true);
                }
            }
            SlotState::Copied => {
                // The slot is being copied or has copied to the new map, so we try
                // to help the resize, and set the third returned value as `true`.
                self.help_resize(outer_map, guard);
                return (result, false, true);
            }
            SlotState::NullOrKey => {
                // There is not special flag on the slot, so we compare the keys.
                if let Some(pair) = entry0 {
                    if key.eq(pair.key.borrow()) {
                        result.tbl_idx = Some(0);
                        // We successfully match the searched key, but we cannot return here,
                        // because we may have duplicated keys in both slots.
                        // We must do the deduplication in this method.
                    }
                }
            }
        }
        // Check the second table.
        let slot1 = self.get_slot(slot_idx1, guard);
        let (_, entry1, state1) = Self::unwrap_slot(slot1);
        match state1 {
            SlotState::Reloc => {
                if self.help_relocate(slot_idx1, false, true, outer_map, guard) {
                    return (result, true, false);
                } else {
                    return (result, false, true);
                }
            }
            SlotState::Copied => {
                self.help_resize(outer_map, guard);
                return (result, false, true);
            }
            SlotState::NullOrKey => {
                if let Some(pair) = entry1 {
                    if key.eq(pair.key.borrow()) {
                        if result.tbl_idx.is_some() {
                            // We have a duplicated key in both slots,
                            // try to delete the second one.
                            self.del_dup(slot_idx0, slot_idx1, outer_map, guard);
                        } else {
                            // Otherwise, we successfully find the key in the
                            // second slot, we can return the table index as 1 then.
                            result.tbl_idx = Some(1);
                        }
                    }
                }
            }
        }

        result.slot0 = slot0;
        result.slot1 = slot1;
        (result, false, false)
    }

    /// `help_relocate` helps relocate the slot at `src_idx` to the other corresponding slot.
    /// It will first mark the `src_slot`'s state as `Reloc` (when the caller is the initiator),
    /// and then try to copy the `src_slot` to the `dst_slot` if `dst_slot` is empty. But if
    ///  `dst_slot` is not empty, this method will do nothing.
    /// This method may fail because one of the slot has been copied to the new map. If so,
    /// this method returns false, otherwise returns true.
    #[allow(clippy::too_many_lines)]
    fn help_relocate(
        &self,
        src_idx: SlotIndex,
        initiator: bool,
        need_help_resize: bool,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) -> bool {
        loop {
            let mut src_slot = self.get_slot(src_idx, guard);
            while initiator && Self::slot_state(src_slot) != SlotState::Reloc {
                if Self::slot_state(src_slot) == SlotState::Copied {
                    if need_help_resize {
                        self.help_resize(outer_map, guard);
                    }
                    return false;
                }
                if Self::slot_is_empty(src_slot) {
                    return true;
                }
                let new_slot_with_reloc = src_slot.with_lower_u2(SlotState::Reloc.into_u8());
                // Only this CAS can set the `Relocation` state!
                if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                    .compare_and_set(src_slot, new_slot_with_reloc, Ordering::SeqCst, guard)
                    .is_ok()
                {
                    // do nothing here, the slot state is checked by the while condition.
                }
                src_slot = self.get_slot(src_idx, guard);
            }

            let (src_count, src_entry, src_state) = Self::unwrap_slot(src_slot);
            match src_state {
                SlotState::NullOrKey => {
                    return true;
                }
                SlotState::Copied => {
                    if need_help_resize {
                        self.help_resize(outer_map, guard);
                    }
                    return false;
                }
                SlotState::Reloc => {}
            }
            let src_key = if let Some(pair) = src_entry {
                &pair.key
            } else {
                // The slot is empty, no need for a relocation.
                return true;
            };
            let dst_idx = self.get_index(1_usize.overflow_sub(src_idx.tbl_idx), src_key);
            let dst_slot = self.get_slot(dst_idx, guard);
            let (dst_count, dst_entry, dst_state) = Self::unwrap_slot(dst_slot);
            if let SlotState::Copied = dst_state {
                let new_slot_without_mark = src_slot.with_lower_u2(SlotState::NullOrKey.into_u8());
                self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                    .compare_and_set(src_slot, new_slot_without_mark, Ordering::SeqCst, guard)
                    .ok();
                if need_help_resize {
                    self.help_resize(outer_map, guard);
                }
                return false;
            }
            if dst_entry.is_none() {
                // overflow will be handled by `check_counter`.
                let new_count = if src_count > dst_count {
                    src_count.overflow_add(1)
                } else {
                    dst_count.overflow_add(1)
                };
                if self.get_slot(src_idx, guard).as_raw() != src_slot.as_raw() {
                    continue;
                }
                let new_slot = Self::set_rlcount(src_slot, new_count, guard)
                    .with_lower_u2(SlotState::NullOrKey.into_u8());

                if self.tables[dst_idx.tbl_idx][dst_idx.slot_idx]
                    .compare_and_set(dst_slot, new_slot, Ordering::SeqCst, guard)
                    .is_ok()
                {
                    // overflow will be handled by `check_counter`.
                    let empty_slot =
                        Self::set_rlcount(SharedPtr::null(), src_count.overflow_add(1), guard);
                    if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                        .compare_and_set(src_slot, empty_slot, Ordering::SeqCst, guard)
                        .is_ok()
                    {
                        // do nothing
                    }
                    return true;
                }
            }
            // dst is not null
            if src_slot.as_raw() == dst_slot.as_raw()
                && Self::slot_state(dst_slot) != SlotState::Reloc
            {
                // overflow will be handled by `check_counter`.
                let empty_slot =
                    Self::set_rlcount(SharedPtr::null(), src_count.overflow_add(1), guard);
                if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                    .compare_and_set(src_slot, empty_slot, Ordering::SeqCst, guard)
                    .is_ok()
                {
                    // do nothing
                }
                return true;
            }
            // overflow will be handled by `check_counter`.
            let new_slot_without_mark =
                Self::set_rlcount(src_slot, src_count.overflow_add(1), guard)
                    .with_lower_u2(SlotState::NullOrKey.into_u8());
            if self.tables[src_idx.tbl_idx][src_idx.slot_idx]
                .compare_and_set(src_slot, new_slot_without_mark, Ordering::SeqCst, guard)
                .is_ok()
            {
                // do nothing
            }
            return true;
        }
    }

    /// `resize` resizes the table.
    fn resize(&self, outer_map: &AtomicPtr<Self>, guard: &'guard Guard) {
        if self
            .new_map
            .load(Ordering::SeqCst, guard)
            .as_raw()
            .is_null()
        {
            let new_capacity = self.capacity().saturating_mul(2);
            // Allocate the new map.
            let new_map = SharedPtr::from_box(Box::new(Self::with_capacity(
                new_capacity,
                [self.hash_builders[0].clone(), self.hash_builders[1].clone()],
            )));
            // Initialize `self.new_map`.
            if self
                .new_map
                .compare_and_set(SharedPtr::null(), new_map, Ordering::SeqCst, guard)
                .is_err()
            {
                // Free the box
                // TODO: we can avoid this redundent allocation.
                unsafe {
                    drop(new_map.into_box());
                }
            }
        }
        self.help_resize(outer_map, guard);
    }

    /// `help_resize` helps copy the current `MapInner` into `self.new_map`.
    fn help_resize(&self, outer_map: &AtomicPtr<Self>, guard: &'guard Guard) {
        let capacity = self.capacity();
        let capacity_per_table = self.tables[0].len();
        loop {
            let next_copy_idx = self.next_copy_idx.fetch_add(1, Ordering::SeqCst);
            if next_copy_idx >= capacity {
                break;
            }
            let slot_idx = SlotIndex {
                // overflow will never happen here.
                tbl_idx: next_copy_idx.overflow_div(capacity_per_table),
                slot_idx: next_copy_idx.overflowing_rem(capacity_per_table).0,
            };
            self.copy_slot(slot_idx, outer_map, guard);
        }

        // waiting for finishing the copy of all the slots.
        // Notice: this is not lock-free, because we use a busy-waiting here.
        loop {
            let copied_num = self.copied_num.load(Ordering::SeqCst);
            if copied_num == capacity {
                // try to promote the new map
                let current_map = SharedPtr::from_raw(self);
                let new_map = self.new_map.load(Ordering::SeqCst, guard);
                if let Ok(current_map) =
                    outer_map.compare_and_set(current_map, new_map, Ordering::SeqCst, guard)
                {
                    unsafe {
                        guard.defer_unchecked(move || {
                            drop(current_map.into_box());
                        });
                    }
                }
                break;
            }
        }
    }

    /// `copy_slot` copies a single slot into the new map.
    fn copy_slot(&self, slot_idx: SlotIndex, outer_map: &AtomicPtr<Self>, guard: &'guard Guard) {
        loop {
            let slot = self.get_slot(slot_idx, guard);
            let (_, _, state) = Self::unwrap_slot(slot);
            match state {
                SlotState::NullOrKey => {
                    let new_slot = slot.with_lower_u2(SlotState::Copied.into_u8());
                    match self.tables[slot_idx.tbl_idx][slot_idx.slot_idx].compare_and_set(
                        slot,
                        new_slot,
                        Ordering::SeqCst,
                        guard,
                    ) {
                        Ok(_) => {
                            if !Self::slot_is_empty(new_slot) {
                                // The insert might fail because the new_map is rezied.
                                // If so, we need to automically re-load the new_map (which has been resized)
                                // and try the insert again.
                                loop {
                                    let new_map = if let Some(new_inner) = unsafe {
                                        self.new_map.load(Ordering::SeqCst, guard).as_raw().as_ref()
                                    } {
                                        new_inner
                                    } else {
                                        // should never be here.
                                        return;
                                    };
                                    if let InsertResult::Retry = new_map.insert(
                                        SharedPtr::from_raw(slot.as_raw()),
                                        InsertType::InsertOrReplace,
                                        &self.new_map,
                                        guard,
                                    ) {
                                        continue;
                                    }
                                    break;
                                }
                            }
                            self.copied_num.fetch_add(1, Ordering::SeqCst);
                            break;
                        }
                        Err(_) => continue,
                    }
                }
                SlotState::Reloc => {
                    self.help_relocate(slot_idx, false, false, outer_map, guard);
                    continue;
                }
                SlotState::Copied => {
                    // shoule never be here
                }
            }
        }
    }

    /// `relocate` tries to make the slot in `origin_idx` empty, in order to insert
    /// a new key-value pair into it.
    fn relocate(
        &self,
        origin_idx: SlotIndex,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) -> RelocateResult {
        let threshold = self.relocation_threshold();
        let mut route = Vec::with_capacity(10); // TODO: optimize this.
        let mut start_level = 0;
        let mut slot_idx = origin_idx;

        // This method consists of two steps:
        // 1. Path Discovery
        //    This step aims to find the cuckoo path which ends with an empty slot,
        //    so we could swap the empty slot backward to the `origin_idx`. Once the
        //    slot at `origin_idx` is empty, the new key-value pair can be inserted.
        // 2. Swap slot
        //    When we have discovered a cuckoo path, we can swap the empty slot backward
        //    to the slot at `origin_idx`.

        'main_loop: loop {
            let mut found = false;
            let mut depth = start_level;
            loop {
                let mut slot = self.get_slot(slot_idx, guard);
                while Self::slot_state(slot) == SlotState::Reloc {
                    if !self.help_relocate(slot_idx, false, true, outer_map, guard) {
                        return RelocateResult::Resized;
                    }
                    slot = self.get_slot(slot_idx, guard);
                }
                let (_, entry, state) = Self::unwrap_slot(slot);
                if let SlotState::Copied = state {
                    self.help_resize(outer_map, guard);
                    return RelocateResult::Resized;
                }
                if let Some(entry) = entry {
                    let key = &entry.key;

                    // If there are duplicated keys in both slots, we may
                    // meet an endless loop. So we must do the dedup here.
                    let next_slot_idx = self.get_index(1_usize.overflow_sub(slot_idx.tbl_idx), key);
                    let next_slot = self.get_slot(next_slot_idx, guard);
                    let (_, next_entry, next_state) = Self::unwrap_slot(next_slot);
                    if let SlotState::Copied = next_state {
                        self.help_resize(outer_map, guard);
                        return RelocateResult::Resized;
                    }
                    if let Some(pair) = next_entry {
                        if pair.key.eq(key) {
                            if slot_idx.tbl_idx == 0 {
                                self.del_dup(slot_idx, next_slot_idx, outer_map, guard);
                            } else {
                                self.del_dup(next_slot_idx, slot_idx, outer_map, guard);
                            }
                        }
                    }

                    // push the slot into the cuckoo path.
                    if route.len() <= depth {
                        route.push(slot_idx);
                    } else {
                        route[depth] = slot_idx;
                    }
                    slot_idx = next_slot_idx;
                } else {
                    found = true;
                }
                depth = depth.overflow_add(1);
                if found || depth >= threshold {
                    break;
                }
            }

            if found {
                depth = depth.overflow_sub(1);
                'slot_swap: for i in (0..depth).rev() {
                    let src_idx = route[i];
                    let mut src_slot = self.get_slot(src_idx, guard);
                    while Self::slot_state(src_slot) == SlotState::Reloc {
                        if !self.help_relocate(src_idx, false, true, outer_map, guard) {
                            return RelocateResult::Resized;
                        }
                        src_slot = self.get_slot(src_idx, guard);
                    }
                    let (_, entry, state) = Self::unwrap_slot(src_slot);
                    if let SlotState::Copied = state {
                        self.help_resize(outer_map, guard);
                        return RelocateResult::Resized;
                    }
                    if let Some(pair) = entry {
                        let dst_idx =
                            self.get_index(1_usize.overflow_sub(src_idx.tbl_idx), &pair.key);
                        let (_, dst_entry, dst_state) = self.get_entry(dst_idx, guard);
                        if let SlotState::Copied = dst_state {
                            self.help_resize(outer_map, guard);
                            return RelocateResult::Resized;
                        }
                        // `dst_entry` should be empty. If it is not, it mains the cuckoo path
                        // has been changed by other threads. Go back to complete the path.
                        if dst_entry.is_some() {
                            start_level = i.overflow_add(1);
                            slot_idx = dst_idx;
                            continue 'main_loop;
                        }
                        if !self.help_relocate(src_idx, true, true, outer_map, guard) {
                            return RelocateResult::Resized;
                        }
                    }
                    continue 'slot_swap;
                }
                return RelocateResult::Succ;
            }
            return RelocateResult::NeedResize;
        }
    }

    /// `del_dup` deletes the duplicated key. It only deletes the key in the second table.
    fn del_dup(
        &self,
        slot_idx0: SlotIndex,
        slot_idx1: SlotIndex,
        outer_map: &AtomicPtr<Self>,
        guard: &'guard Guard,
    ) {
        let slot0 = self.get_slot(slot_idx0, guard);
        let slot1 = self.get_slot(slot_idx1, guard);

        if Self::slot_state(slot0) == SlotState::Reloc {
            self.help_relocate(slot_idx0, false, false, outer_map, guard);
            return;
        }
        if Self::slot_state(slot0) == SlotState::Copied {
            return;
        }
        if Self::slot_state(slot1) == SlotState::Reloc {
            self.help_relocate(slot_idx1, false, false, outer_map, guard);
            return;
        }
        if Self::slot_state(slot1) == SlotState::Copied {
            return;
        }

        if slot1.as_raw() == slot0.as_raw() {
            // FIXME:
            //   This is a tricky fix for the duplicated key problem which
            //   cannot deduplicate the co-existed key (with the same pointer).
            //   This kind of duplicated key is generated by `help_relocate`.
            //   We hope another `help_relocate` or `resize` can solve it.
            return;
        }

        let (_, entry0, _) = Self::unwrap_slot(slot0);
        let (slot1_count, entry1, _) = Self::unwrap_slot(slot1);
        let mut need_dedup = false;
        if let Some(pair0) = entry0 {
            if let Some(pair1) = entry1 {
                need_dedup = pair0.key.eq(&pair1.key);
            }
        }
        if !need_dedup {
            return;
        }
        let need_free = slot0.as_raw() != slot1.as_raw();
        let empty_slot = Self::set_rlcount(SharedPtr::null(), slot1_count, guard);
        if let Ok(old_slot) = self.tables[slot_idx1.tbl_idx][slot_idx1.slot_idx].compare_and_set(
            slot1,
            empty_slot,
            Ordering::SeqCst,
            guard,
        ) {
            if need_free {
                self.size.fetch_sub(1, Ordering::SeqCst);
                Self::defer_drop_ifneed(old_slot, guard);
            }
        }
    }

    /// `check_counter` checks the relocation count to decide
    /// whether we need to read the slots again.
    fn check_counter(c00_u8: u8, c01_u8: u8, c10_u8: u8, c11_u8: u8) -> bool {
        // Normally, the checked condition should be:
        //   c10 >= c00 + 2 && c11 >= c01 + 2 && c11 >= c00 + 3
        // But the relocation count might overflow. If so, we return true.

        // FIXME:
        // This method is not really safe for the overflow.
        // For example,
        // c00_u8 = 1          |
        //                     |   slot has been relocated many times,
        // c11_u8 = 257%256    |
        //        = 1          |
        //
        // As a result, we cannot detect the overflow.
        // There is a solution for this problem:
        // We use the highest bit of the counter as the overflow flag.
        // The flag will be marked as 1 if the overflow happens.
        // So we can detect the overflow in this method and reset the flag.
        // And no matter how many times the overflow happends, we can detect it.

        let (c00, c01, c10, c11) = (
            c00_u8.cast::<u16>(),
            c01_u8.cast::<u16>(),
            c10_u8.cast::<u16>(),
            c11_u8.cast::<u16>(),
        );
        (c10 < c00)
            || (c11 < c01)
            || (c10 >= c00.overflow_add(2)
                && c11 >= c01.overflow_add(2)
                && c11 >= c00.overflow_add(3))
    }

    /// `drop_entries` drops the entries.
    pub fn drop_entries(&self, guard: &'guard Guard) {
        for i in 0..2 {
            for j in 0..self.tables[0].len() {
                // key might be duplicated, so we only free the one in primary table.
                let slot = self.get_slot(
                    SlotIndex {
                        tbl_idx: i,
                        slot_idx: j,
                    },
                    guard,
                );
                if i == 1 {
                    let (_, entry, _) = Self::unwrap_slot(slot);
                    if let Some(pair) = entry {
                        let primary_slot_idx = self.get_index(0, &pair.key);
                        let primary_slot = self.get_slot(primary_slot_idx, guard);
                        if primary_slot.as_raw() == slot.as_raw() {
                            continue;
                        }
                    }
                }
                Self::defer_drop_ifneed(slot, guard);
            }
        }
    }

    /// `relocation_threshold` returns the threshold of triggering resize.
    fn relocation_threshold(&self) -> usize {
        self.tables[0].len()
    }

    /// `slot_state` returns the state of the slot.
    fn slot_state(slot: SharedPtr<'guard, KVPair<K, V>>) -> SlotState {
        let (_, _, lower_u2) = slot.decompose();
        SlotState::from_u8(lower_u2)
    }

    /// `slot_is_empty` checks if the slot is a null pointer.
    fn slot_is_empty(slot: SharedPtr<'guard, KVPair<K, V>>) -> bool {
        let raw = slot.as_raw();
        raw.is_null()
    }

    /// `unwrap_slot` unwraps the slot into three parts:
    /// 1. the relocation count
    /// 2. the key value pair
    /// 3. the state of the slot
    fn unwrap_slot(
        slot: SharedPtr<'guard, KVPair<K, V>>,
    ) -> (u8, Option<&'guard KVPair<K, V>>, SlotState) {
        let (rlcount, raw, lower_u2) = slot.decompose();
        let state = SlotState::from_u8(lower_u2);
        unsafe { (rlcount, raw.as_ref(), state) }
    }

    /// `set_rlcount` sets the relocation count of a slot.
    fn set_rlcount(
        slot: SharedPtr<'guard, KVPair<K, V>>,
        rlcount: u8,
        _: &'guard Guard,
    ) -> SharedPtr<'guard, KVPair<K, V>> {
        slot.with_higher_u8(rlcount)
    }

    /// `get_rlcount` returns the relocation count of a slot.
    fn get_rlcount(slot: SharedPtr<'guard, KVPair<K, V>>) -> u8 {
        let (rlcount, _, _) = slot.decompose();
        rlcount
    }

    /// `get_entry` atomically loads the slot and unwrap it.
    fn get_entry(
        &self,
        slot_idx: SlotIndex,
        guard: &'guard Guard,
    ) -> (u8, Option<&'guard KVPair<K, V>>, SlotState) {
        // TODO: split this method by different memory ordering.
        Self::unwrap_slot(self.get_slot(slot_idx, guard))
    }

    /// `get_slot` atomically loads the slot.
    fn get_slot(
        &self,
        slot_idx: SlotIndex,
        guard: &'guard Guard,
    ) -> SharedPtr<'guard, KVPair<K, V>> {
        self.tables[slot_idx.tbl_idx][slot_idx.slot_idx].load(Ordering::SeqCst, guard)
    }

    /// `get_index` hashes the key and return the slot index.
    fn get_index<Q: Hash + ?Sized>(&self, tbl_idx: usize, key: &Q) -> SlotIndex {
        let mut hasher = self.hash_builders[tbl_idx].build_hasher();
        key.hash(&mut hasher);
        let hash_value = hasher.finish().cast::<usize>();
        // The conversion from u64 to usize will never fail in a 64-bit env.
        // self.tables[0].len() is always non-zero, so the arithmetic is safe here.
        let slot_idx = hash_value.overflowing_rem(self.tables[0].len()).0;
        SlotIndex { tbl_idx, slot_idx }
    }

    /// `defer_drop_ifneed` tries to defer to drop the slot if not empty.
    fn defer_drop_ifneed(slot: SharedPtr<'guard, KVPair<K, V>>, guard: &'guard Guard) {
        if !Self::slot_is_empty(slot) {
            unsafe {
                // We take over the ownership here.
                // Because only one thread can call this method for the same
                // kv-pair, only one thread can take the ownership.
                guard.defer_unchecked(move || {
                    drop(slot.into_box());
                });
            }
        }
    }
}
