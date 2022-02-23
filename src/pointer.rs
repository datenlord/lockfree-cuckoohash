use clippy_utilities::{Cast, OverflowArithmetic};
use crossbeam_epoch::Guard;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `AtomicPtr` is a pointer which can only be manipulated by
/// atomic operations.
#[derive(Debug)]
pub struct AtomicPtr<T: ?Sized> {
    /// `data` is the value of the atomic pointer.
    data: AtomicUsize,
    /// used for type inference.
    _marker: PhantomData<*mut T>,
}

unsafe impl<T: ?Sized + Send + Sync> Send for AtomicPtr<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for AtomicPtr<T> {}

impl<T> AtomicPtr<T> {
    /// `from_usize` creates an `AtomicPtr` from `usize`.
    const fn from_usize(data: usize) -> Self {
        Self {
            data: AtomicUsize::new(data),
            _marker: PhantomData,
        }
    }

    /// `new` creates the `value` on heap and returns its `AtomicPtr`.
    #[allow(clippy::as_conversions)]
    pub fn new(value: T) -> Self {
        let b = Box::new(value);
        let raw_ptr = Box::into_raw(b);
        Self::from_usize(raw_ptr as usize)
    }

    /// `null` returns a `AtomicPtr` with a null pointer.
    pub const fn null() -> Self {
        Self::from_usize(0)
    }

    /// `load` atomically loads the pointer.
    pub fn load<'g>(&self, ord: Ordering, _: &'g Guard) -> SharedPtr<'g, T> {
        SharedPtr::from_usize(self.data.load(ord))
    }

    /// `compare_and_set` wraps the `compare_exchange` method of `AtomicUsize`.
    pub fn compare_and_set<'g>(
        &self,
        current_ptr: SharedPtr<'_, T>,
        new_ptr: SharedPtr<'_, T>,
        ord: Ordering,
        _: &'g Guard,
    ) -> Result<SharedPtr<'g, T>, (SharedPtr<'g, T>, SharedPtr<'g, T>)> {
        let new = new_ptr.as_usize();
        // TODO: different ordering.
        self.data
            .compare_exchange(current_ptr.as_usize(), new, ord, ord)
            .map(|_| SharedPtr::from_usize(current_ptr.as_usize()))
            .map_err(|current| (SharedPtr::from_usize(current), SharedPtr::from_usize(new)))
    }
}

/// `SharedPtr` is a pointer which can be shared by multi-threads.
/// `SharedPtr` can only be used with 64bit-wide pointer, and the
/// pointer address must be 4-byte aligned.
pub struct SharedPtr<'g, T: 'g> {
    /// `data` is the value of the pointers.
    /// It will be spilt into three parts:
    /// [higher_u8, raw_pointer, lower_u2]
    ///    8 bits     54 bits     2 bits
    data: usize,
    /// used for type inference.
    _marker: PhantomData<(&'g (), *const T)>,
}

impl<T> Clone for SharedPtr<'_, T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for SharedPtr<'_, T> {}

impl<T> SharedPtr<'_, T> {
    /// `from_usize` creates a `SharedPtr` from a `usize`.
    pub const fn from_usize(data: usize) -> Self {
        SharedPtr {
            data,
            _marker: PhantomData,
        }
    }

    /// `from_box` creates a `SharedPtr` from a `Box`.
    pub fn from_box(b: Box<T>) -> Self {
        Self::from_raw(Box::into_raw(b))
    }

    /// `from_raw` creates a `SharedPtr` from a raw pointer.
    #[allow(clippy::as_conversions)]
    pub fn from_raw(raw: *const T) -> Self {
        Self::from_usize(raw as usize)
    }

    /// `null` returns a null `SharedPtr`.
    pub const fn null() -> Self {
        Self::from_usize(0)
    }

    /// `into_box` converts the pointer into a Box<T>.
    #[must_use]
    pub unsafe fn into_box(self) -> Box<T> {
        Box::from_raw(self.as_mut_raw())
    }

    /// `as_usize` converts the pointer to `usize`.
    pub const fn as_usize(self) -> usize {
        self.data
    }

    /// `decompose_lower_u2` decomposes the pointer into two parts:
    /// 1. the higher 62 bits
    /// 2. the lower 2 bits
    fn decompose_lower_u2(data: usize) -> (usize, u8) {
        let mask: usize = 3;
        // The unwrap is safe here, because we have mask the lower 2 bits.
        (data & !mask, (data & mask).cast::<u8>())
    }

    /// `decompose_higher_u8` decomposes the pointer into two parts:
    /// 1. the higher 8 bits
    /// 2. the lower 56 bits
    fn decompose_higher_u8(data: usize) -> (u8, usize) {
        let mask: usize = (1_usize.overflowing_shl(56).0).overflow_sub(1);
        // The conversion is safe here, because we have shifted 56 bits.
        (data.overflow_shr(56).cast::<u8>(), data & mask)
    }

    /// `decompose` decomposes the pointer into three parts:
    /// 1. the higher 8 bits
    /// 2. the raw pointer
    /// 3. the lower 2 bits
    #[allow(clippy::as_conversions)]
    pub fn decompose(self) -> (u8, *const T, u8) {
        let data = self.data;
        let (higher_u62, lower_u2) = Self::decompose_lower_u2(data);
        let (higher_u8, raw_ptr) = Self::decompose_higher_u8(higher_u62);
        (higher_u8, raw_ptr as *const T, lower_u2)
    }

    /// `as_raw` extracts the raw pointer.
    pub fn as_raw(self) -> *const T {
        let (_, raw, _) = self.decompose();
        raw
    }

    /// `as_mut_raw` extracts the mutable raw pointer.
    #[allow(clippy::as_conversions)]
    pub fn as_mut_raw(self) -> *mut T {
        let const_raw = self.as_raw();
        const_raw as *mut T
    }

    /// `with_lower_u2` resets the lower 2 bits of the pointer.
    pub fn with_lower_u2(self, lower_u8: u8) -> Self {
        let mask: usize = 3;
        // Convert a u8 to usize is always safe.
        Self::from_usize(self.data & !mask | lower_u8.cast::<usize>())
    }

    /// `with_higher_u8` resets the higher 8 bits of pointer.
    pub fn with_higher_u8(self, higher_u8: u8) -> Self {
        let data = self.data;
        let mask: usize = (1_usize.overflowing_shl(56).0).overflow_sub(1);
        // Convert a u8 to usize is always safe.
        Self::from_usize((data & mask) | ((higher_u8.cast::<usize>()).overflowing_shl(56).0))
    }
}
