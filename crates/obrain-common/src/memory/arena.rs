//! Epoch-based arena allocator for MVCC.
//!
//! This is how Obrain manages memory for versioned data. Each epoch gets its
//! own arena, and when all readers from an old epoch finish, we free the whole
//! thing at once. Much faster than tracking individual allocations.
//!
//! Use [`ArenaAllocator`] to manage multiple epochs, or [`Arena`] directly
//! if you're working with a single epoch.

// Arena allocators require unsafe code for memory management
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc, dealloc};
use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;

use crate::types::EpochId;

/// Default chunk size for arena allocations (1 MB).
///
/// Without tiered-storage, arenas use small chunks since records are stored
/// in simple HashMaps. Chunks grow as needed via `alloc()`.
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Chunk size for tiered-storage arenas (8 GB).
///
/// With tiered-storage, `alloc_value_with_offset` stores records in chunked
/// arenas addressed by u64 offsets (high 16 bits = chunk index, low 48 bits
/// = offset within chunk). 8 GB per chunk handles ~128M records (at ~64 bytes
/// each). On modern OSes, `alloc` for large sizes uses mmap(MAP_ANON)
/// internally — pages are allocated lazily on first touch, so virtual memory
/// cost is near-zero until data is written.
///
/// If the primary chunk fills up, new chunks are allocated transparently.
/// With 16-bit chunk indices and 48-bit offsets, the theoretical maximum is
/// 65536 chunks × 256 TB per chunk — effectively unlimited.
#[cfg(feature = "tiered-storage")]
const TIERED_CHUNK_SIZE: usize = 8 * 1024 * 1024 * 1024; // 8 GB

/// Errors from arena allocation operations.
#[derive(Debug, Clone)]
pub enum AllocError {
    /// The system allocator returned null (out of memory).
    OutOfMemory,
    /// The requested epoch does not exist.
    EpochNotFound(EpochId),
    /// Arena chunk has insufficient space for the allocation.
    InsufficientSpace,
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "arena allocation failed: out of memory"),
            Self::EpochNotFound(id) => write!(f, "epoch {id} not found in arena allocator"),
            Self::InsufficientSpace => {
                write!(f, "arena chunk has insufficient space for allocation")
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<AllocError> for crate::Error {
    fn from(e: AllocError) -> Self {
        match e {
            AllocError::OutOfMemory | AllocError::InsufficientSpace => {
                crate::Error::Storage(crate::utils::error::StorageError::Full)
            }
            AllocError::EpochNotFound(id) => {
                crate::Error::Internal(format!("epoch {id} not found in arena allocator"))
            }
        }
    }
}

/// A memory chunk in the arena.
struct Chunk {
    /// Pointer to the start of the chunk.
    ptr: NonNull<u8>,
    /// Total capacity of the chunk.
    capacity: usize,
    /// Current allocation offset.
    offset: AtomicUsize,
}

impl Chunk {
    /// Creates a new chunk with the given capacity.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the system allocator fails.
    fn new(capacity: usize) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(capacity, 16).map_err(|_| AllocError::OutOfMemory)?;
        // SAFETY: We're allocating a valid layout
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory)?;

        Ok(Self {
            ptr,
            capacity,
            offset: AtomicUsize::new(0),
        })
    }

    /// Tries to allocate `size` bytes with the given alignment.
    /// Returns None if there's not enough space.
    fn try_alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        self.try_alloc_with_offset(size, align).map(|(_, ptr)| ptr)
    }

    /// Tries to allocate `size` bytes with the given alignment.
    /// Returns (offset, ptr) where offset is the aligned offset within this chunk.
    /// Returns None if there's not enough space.
    fn try_alloc_with_offset(&self, size: usize, align: usize) -> Option<(u64, NonNull<u8>)> {
        loop {
            let current = self.offset.load(Ordering::Relaxed);

            // Calculate aligned offset
            let aligned = (current + align - 1) & !(align - 1);
            let new_offset = aligned + size;

            if new_offset > self.capacity {
                return None;
            }

            // Try to reserve the space
            match self.offset.compare_exchange_weak(
                current,
                new_offset,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: We've reserved this range exclusively
                    let ptr = unsafe { self.ptr.as_ptr().add(aligned) };
                    return Some((aligned as u64, NonNull::new(ptr)?));
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Returns the amount of memory used in this chunk.
    fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 16).expect("Invalid layout");
        // SAFETY: We allocated this memory with the same layout
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

// SAFETY: Chunk uses atomic operations for thread-safe allocation
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

/// A single epoch's memory arena.
///
/// Allocates by bumping a pointer forward - extremely fast. You can't free
/// individual allocations; instead, drop the whole arena when the epoch
/// is no longer needed.
///
/// Thread-safe: multiple threads can allocate concurrently using atomics.
pub struct Arena {
    /// The epoch this arena belongs to.
    epoch: EpochId,
    /// List of memory chunks.
    chunks: RwLock<Vec<Chunk>>,
    /// Default chunk size for new allocations.
    chunk_size: usize,
    /// Total bytes allocated.
    total_allocated: AtomicUsize,
}

impl Arena {
    /// Creates a new arena for the given epoch.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial chunk allocation fails.
    pub fn new(epoch: EpochId) -> Result<Self, AllocError> {
        Self::with_chunk_size(epoch, DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new arena with a custom chunk size.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial chunk allocation fails.
    pub fn with_chunk_size(epoch: EpochId, chunk_size: usize) -> Result<Self, AllocError> {
        let initial_chunk = Chunk::new(chunk_size)?;
        Ok(Self {
            epoch,
            chunks: RwLock::new(vec![initial_chunk]),
            chunk_size,
            total_allocated: AtomicUsize::new(chunk_size),
        })
    }

    /// Returns the epoch this arena belongs to.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Allocates `size` bytes with the given alignment.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if a new chunk is needed and
    /// the system allocator fails.
    pub fn alloc(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
        // First try to allocate from existing chunks
        {
            let chunks = self.chunks.read();
            for chunk in chunks.iter().rev() {
                if let Some(ptr) = chunk.try_alloc(size, align) {
                    return Ok(ptr);
                }
            }
        }

        // Need a new chunk
        self.alloc_new_chunk(size, align)
    }

    /// Allocates a value of type T.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if allocation fails.
    pub fn alloc_value<T>(&self, value: T) -> Result<&mut T, AllocError> {
        let ptr = self.alloc(std::mem::size_of::<T>(), std::mem::align_of::<T>())?;
        // SAFETY: We've allocated the correct size and alignment
        Ok(unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            &mut *typed_ptr
        })
    }

    /// Allocates a slice of values.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if allocation fails.
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Result<&mut [T], AllocError> {
        if values.is_empty() {
            return Ok(&mut []);
        }

        let size = std::mem::size_of::<T>() * values.len();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align)?;

        // SAFETY: We've allocated the correct size and alignment
        Ok(unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), typed_ptr, values.len());
            std::slice::from_raw_parts_mut(typed_ptr, values.len())
        })
    }

    /// Allocates a value and returns a composite u64 offset.
    ///
    /// The offset encodes both the chunk index and the position within that
    /// chunk: `(chunk_index << 48) | offset_in_chunk`.  This allows addressing
    /// across multiple chunks, removing the previous 2 GB / 4 GB ceiling.
    ///
    /// The allocator tries existing chunks (newest first for locality), then
    /// grows a new chunk if none have space.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the system allocator cannot
    /// provide a new chunk.
    #[cfg(feature = "tiered-storage")]
    pub fn alloc_value_with_offset<T>(&self, value: T) -> Result<(u64, &mut T), AllocError> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        // Try to allocate in an existing chunk (newest first for cache locality)
        {
            let chunks = self.chunks.read();
            for (chunk_idx, chunk) in chunks.iter().enumerate().rev() {
                if let Some((offset_in_chunk, ptr)) = chunk.try_alloc_with_offset(size, align) {
                    let composite = Self::encode_offset(chunk_idx, offset_in_chunk);
                    // SAFETY: We've allocated the correct size and alignment
                    return Ok(unsafe {
                        let typed_ptr = ptr.as_ptr().cast::<T>();
                        typed_ptr.write(value);
                        (composite, &mut *typed_ptr)
                    });
                }
            }
        }

        // No existing chunk has space — grow a new one
        let chunk_size = self.chunk_size.max(size + align);
        let new_chunk = Chunk::new(chunk_size)?;
        self.total_allocated
            .fetch_add(chunk_size, Ordering::Relaxed);

        let (offset_in_chunk, ptr) = new_chunk
            .try_alloc_with_offset(size, align)
            .expect("fresh chunk sized to fit");

        let mut chunks = self.chunks.write();
        let chunk_idx = chunks.len();
        chunks.push(new_chunk);

        let composite = Self::encode_offset(chunk_idx, offset_in_chunk);
        // SAFETY: We've allocated the correct size and alignment
        Ok(unsafe {
            let typed_ptr = ptr.as_ptr().cast::<T>();
            typed_ptr.write(value);
            (composite, &mut *typed_ptr)
        })
    }

    /// Encodes a (chunk_index, offset_in_chunk) pair into a single u64.
    /// Layout: bits [63..48] = chunk_index (16 bits), bits [47..0] = offset (48 bits).
    #[cfg(feature = "tiered-storage")]
    #[inline]
    fn encode_offset(chunk_idx: usize, offset_in_chunk: u64) -> u64 {
        debug_assert!(chunk_idx < (1 << 16), "chunk index overflow: {chunk_idx}");
        debug_assert!(
            offset_in_chunk < (1u64 << 48),
            "offset overflow: {offset_in_chunk}"
        );
        ((chunk_idx as u64) << 48) | offset_in_chunk
    }

    /// Decodes a composite u64 offset into (chunk_index, offset_in_chunk).
    #[cfg(feature = "tiered-storage")]
    #[inline]
    fn decode_offset(composite: u64) -> (usize, usize) {
        let chunk_idx = (composite >> 48) as usize;
        let offset = (composite & ((1u64 << 48) - 1)) as usize;
        (chunk_idx, offset)
    }

    /// Reads a value at the given composite offset.
    ///
    /// The offset must have been returned by `alloc_value_with_offset`.
    /// It encodes both the chunk index and the position within that chunk.
    ///
    /// # Safety
    ///
    /// - The offset must have been returned by a previous `alloc_value_with_offset` call
    /// - The type T must match what was stored at that offset
    /// - The arena must not have been dropped
    #[cfg(feature = "tiered-storage")]
    pub unsafe fn read_at<T>(&self, offset: u64) -> &T {
        let (chunk_idx, offset_in_chunk) = Self::decode_offset(offset);
        let chunks = self.chunks.read();
        let chunk = &chunks[chunk_idx];

        debug_assert!(
            offset_in_chunk + std::mem::size_of::<T>() <= chunk.used(),
            "read_at: chunk[{}] offset {} + size_of::<{}>() = {} exceeds chunk used bytes {}",
            chunk_idx,
            offset_in_chunk,
            std::any::type_name::<T>(),
            offset_in_chunk + std::mem::size_of::<T>(),
            chunk.used()
        );
        debug_assert!(
            offset_in_chunk.is_multiple_of(std::mem::align_of::<T>()),
            "read_at: offset {} is not aligned for {} (alignment {})",
            offset_in_chunk,
            std::any::type_name::<T>(),
            std::mem::align_of::<T>()
        );

        // SAFETY: Caller guarantees offset is valid and T matches stored type
        unsafe {
            let ptr = chunk.ptr.as_ptr().add(offset_in_chunk).cast::<T>();
            &*ptr
        }
    }

    /// Reads a value mutably at the given composite offset.
    ///
    /// # Safety
    ///
    /// - The offset must have been returned by a previous `alloc_value_with_offset` call
    /// - The type T must match what was stored at that offset
    /// - The arena must not have been dropped
    /// - No other references to this value may exist
    #[cfg(feature = "tiered-storage")]
    pub unsafe fn read_at_mut<T>(&self, offset: u64) -> &mut T {
        let (chunk_idx, offset_in_chunk) = Self::decode_offset(offset);
        let chunks = self.chunks.read();
        let chunk = &chunks[chunk_idx];

        debug_assert!(
            offset_in_chunk + std::mem::size_of::<T>() <= chunk.capacity,
            "read_at_mut: chunk[{}] offset {} + size_of::<{}>() = {} exceeds chunk capacity {}",
            chunk_idx,
            offset_in_chunk,
            std::any::type_name::<T>(),
            offset_in_chunk + std::mem::size_of::<T>(),
            chunk.capacity
        );
        debug_assert!(
            offset_in_chunk.is_multiple_of(std::mem::align_of::<T>()),
            "read_at_mut: offset {} is not aligned for {} (alignment {})",
            offset_in_chunk,
            std::any::type_name::<T>(),
            std::mem::align_of::<T>()
        );

        // SAFETY: Caller guarantees offset is valid, T matches, and no aliasing
        unsafe {
            let ptr = chunk.ptr.as_ptr().add(offset_in_chunk).cast::<T>();
            &mut *ptr
        }
    }

    /// Allocates a new chunk and performs the allocation.
    fn alloc_new_chunk(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
        let chunk_size = self.chunk_size.max(size + align);
        let chunk = Chunk::new(chunk_size)?;

        self.total_allocated
            .fetch_add(chunk_size, Ordering::Relaxed);

        // The chunk was sized to fit this allocation, so this cannot fail.
        let ptr = chunk
            .try_alloc(size, align)
            .expect("fresh chunk sized to fit");

        let mut chunks = self.chunks.write();
        chunks.push(chunk);

        Ok(ptr)
    }

    /// Returns the total memory allocated by this arena.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Returns the total memory used (not just allocated capacity).
    #[must_use]
    pub fn total_used(&self) -> usize {
        let chunks = self.chunks.read();
        chunks.iter().map(Chunk::used).sum()
    }

    /// Returns statistics about this arena.
    #[must_use]
    pub fn stats(&self) -> ArenaStats {
        let chunks = self.chunks.read();
        ArenaStats {
            epoch: self.epoch,
            chunk_count: chunks.len(),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_used: chunks.iter().map(Chunk::used).sum(),
        }
    }
}

/// Statistics about an arena.
#[derive(Debug, Clone)]
pub struct ArenaStats {
    /// The epoch this arena belongs to.
    pub epoch: EpochId,
    /// Number of chunks allocated.
    pub chunk_count: usize,
    /// Total bytes allocated.
    pub total_allocated: usize,
    /// Total bytes used.
    pub total_used: usize,
}

/// Manages arenas across multiple epochs.
///
/// Use this to create new epochs, allocate in the current epoch, and
/// clean up old epochs when they're no longer needed.
pub struct ArenaAllocator {
    /// Map of epochs to arenas.  Multiple epochs may share the same
    /// `Arc<Arena>` to avoid allocating a huge chunk per epoch (critical
    /// for tiered-storage where each chunk is 8 GB).
    arenas: RwLock<hashbrown::HashMap<EpochId, Arc<Arena>>>,
    /// Current epoch.
    current_epoch: AtomicUsize,
    /// Default chunk size.
    chunk_size: usize,
    /// The arena currently accepting new allocations.  In tiered-storage
    /// mode, `ensure_epoch` clones this `Arc` instead of creating a
    /// brand-new 2 GB chunk for every epoch.  When the active arena is
    /// full, a new one is created and promoted.
    #[cfg(feature = "tiered-storage")]
    active_write_arena: parking_lot::Mutex<Arc<Arena>>,
}

impl ArenaAllocator {
    /// Creates a new arena allocator.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial arena allocation fails.
    pub fn new() -> Result<Self, AllocError> {
        #[cfg(feature = "tiered-storage")]
        {
            Self::with_chunk_size(TIERED_CHUNK_SIZE)
        }
        #[cfg(not(feature = "tiered-storage"))]
        {
            Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
        }
    }

    /// Creates a new arena allocator with a custom chunk size.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial arena allocation fails.
    pub fn with_chunk_size(chunk_size: usize) -> Result<Self, AllocError> {
        let epoch = EpochId::INITIAL;
        let initial_arena = Arc::new(Arena::with_chunk_size(epoch, chunk_size)?);

        let mut map = hashbrown::HashMap::new();
        map.insert(epoch, Arc::clone(&initial_arena));

        Ok(Self {
            arenas: RwLock::new(map),
            current_epoch: AtomicUsize::new(0),
            chunk_size,
            #[cfg(feature = "tiered-storage")]
            active_write_arena: parking_lot::Mutex::new(initial_arena),
        })
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire) as u64)
    }

    /// Creates a new epoch and returns its ID.
    ///
    /// Each call allocates a **fresh** arena (not shared).  For
    /// tiered-storage write paths prefer [`Self::arena_or_create`] which
    /// coalesces epochs into a shared arena.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the arena allocation fails.
    pub fn new_epoch(&self) -> Result<EpochId, AllocError> {
        let new_id = self.current_epoch.fetch_add(1, Ordering::AcqRel) as u64 + 1;
        let epoch = EpochId::new(new_id);

        let arena = Arc::new(Arena::with_chunk_size(epoch, self.chunk_size)?);
        self.arenas.write().insert(epoch, arena);

        Ok(epoch)
    }

    /// Gets the arena for a specific epoch.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::EpochNotFound` if the epoch doesn't exist.
    pub fn arena(
        &self,
        epoch: EpochId,
    ) -> Result<impl std::ops::Deref<Target = Arena> + '_, AllocError> {
        let arenas = self.arenas.read();
        if !arenas.contains_key(&epoch) {
            return Err(AllocError::EpochNotFound(epoch));
        }
        Ok(parking_lot::RwLockReadGuard::map(arenas, |arenas| {
            arenas[&epoch].as_ref()
        }))
    }

    /// Ensures an arena exists for the given epoch, creating it if necessary.
    /// Returns whether a new arena was created.
    ///
    /// Multiple epochs **share** the same physical arena to avoid
    /// allocating an 8 GB mmap per auto-committed transaction.  A fresh
    /// arena is only created when the active one is full.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if a new arena allocation fails.
    #[cfg(feature = "tiered-storage")]
    pub fn ensure_epoch(&self, epoch: EpochId) -> Result<bool, AllocError> {
        // Fast path: check if epoch already exists
        {
            let arenas = self.arenas.read();
            if arenas.contains_key(&epoch) {
                return Ok(false);
            }
        }

        // Slow path: register the epoch with a shared arena
        let mut arenas = self.arenas.write();
        // Double-check after acquiring write lock
        if arenas.contains_key(&epoch) {
            return Ok(false);
        }

        // Reuse the active write arena instead of creating a new one.
        // Only create a fresh arena when the active one is more than
        // 75% full (extremely unlikely with 2 GB chunks).
        let shared = {
            let mut active = self.active_write_arena.lock();
            let used = active.total_used();
            let capacity = active.total_allocated();
            if used > capacity * 3 / 4 {
                // Active arena is nearly full — promote a new one
                let new_arena = Arc::new(Arena::with_chunk_size(epoch, self.chunk_size)?);
                *active = Arc::clone(&new_arena);
                new_arena
            } else {
                Arc::clone(&active)
            }
        };

        arenas.insert(epoch, shared);
        Ok(true)
    }

    /// Gets or creates an arena for a specific epoch.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` if the arena allocation fails.
    #[cfg(feature = "tiered-storage")]
    pub fn arena_or_create(
        &self,
        epoch: EpochId,
    ) -> Result<impl std::ops::Deref<Target = Arena> + '_, AllocError> {
        self.ensure_epoch(epoch)?;
        self.arena(epoch)
    }

    /// Allocates in the current epoch.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` if allocation fails.
    pub fn alloc(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
        let epoch = self.current_epoch();
        let arenas = self.arenas.read();
        arenas
            .get(&epoch)
            .expect("current epoch always exists")
            .alloc(size, align)
    }

    /// Drops an epoch, freeing all its memory.
    ///
    /// This should only be called when no readers are using this epoch.
    pub fn drop_epoch(&self, epoch: EpochId) {
        self.arenas.write().remove(&epoch);
    }

    /// Returns total memory allocated across all epochs.
    ///
    /// When multiple epochs share the same arena (tiered-storage
    /// coalescing), each physical arena is counted only once.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        let arenas = self.arenas.read();
        let mut seen = hashbrown::HashSet::new();
        arenas
            .values()
            .filter(|a| seen.insert(Arc::as_ptr(a) as usize))
            .map(|a| a.total_allocated())
            .sum()
    }
}

impl Default for ArenaAllocator {
    /// Creates a default arena allocator.
    ///
    /// # Panics
    ///
    /// Panics if the initial arena allocation fails (out of memory).
    fn default() -> Self {
        Self::new().expect("failed to allocate default arena")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic_allocation() {
        let arena = Arena::new(EpochId::INITIAL).unwrap();

        // Allocate some bytes
        let ptr1 = arena.alloc(100, 8).unwrap();
        let ptr2 = arena.alloc(200, 8).unwrap();

        // Pointers should be different
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_arena_value_allocation() {
        let arena = Arena::new(EpochId::INITIAL).unwrap();

        let value = arena.alloc_value(42u64).unwrap();
        assert_eq!(*value, 42);

        *value = 100;
        assert_eq!(*value, 100);
    }

    #[test]
    fn test_arena_slice_allocation() {
        let arena = Arena::new(EpochId::INITIAL).unwrap();

        let slice = arena.alloc_slice(&[1u32, 2, 3, 4, 5]).unwrap();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        slice[0] = 10;
        assert_eq!(slice[0], 10);
    }

    #[test]
    fn test_arena_large_allocation() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 1024).unwrap();

        // Allocate something larger than the chunk size
        let _ptr = arena.alloc(2048, 8).unwrap();

        // Should have created a new chunk
        assert!(arena.stats().chunk_count >= 2);
    }

    #[test]
    fn test_arena_allocator_epochs() {
        let allocator = ArenaAllocator::new().unwrap();

        let epoch0 = allocator.current_epoch();
        assert_eq!(epoch0, EpochId::INITIAL);

        let epoch1 = allocator.new_epoch().unwrap();
        assert_eq!(epoch1, EpochId::new(1));

        let epoch2 = allocator.new_epoch().unwrap();
        assert_eq!(epoch2, EpochId::new(2));

        // Current epoch should be the latest
        assert_eq!(allocator.current_epoch(), epoch2);
    }

    #[test]
    fn test_arena_allocator_allocation() {
        let allocator = ArenaAllocator::new().unwrap();

        let ptr1 = allocator.alloc(100, 8).unwrap();
        let ptr2 = allocator.alloc(100, 8).unwrap();

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_arena_drop_epoch() {
        let allocator = ArenaAllocator::new().unwrap();

        let initial_mem = allocator.total_allocated();

        let epoch1 = allocator.new_epoch().unwrap();
        // Allocate some memory in the new epoch
        {
            let arena = allocator.arena(epoch1).unwrap();
            arena.alloc(10000, 8).unwrap();
        }

        let after_alloc = allocator.total_allocated();
        assert!(after_alloc > initial_mem);

        // Drop the epoch
        allocator.drop_epoch(epoch1);

        // Memory should decrease
        let after_drop = allocator.total_allocated();
        assert!(after_drop < after_alloc);
    }

    #[test]
    fn test_arena_stats() {
        let arena = Arena::with_chunk_size(EpochId::new(5), 4096).unwrap();

        let stats = arena.stats();
        assert_eq!(stats.epoch, EpochId::new(5));
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.total_allocated, 4096);
        assert_eq!(stats.total_used, 0);

        arena.alloc(100, 8).unwrap();
        let stats = arena.stats();
        assert!(stats.total_used >= 100);
    }
}

#[cfg(all(test, feature = "tiered-storage"))]
mod tiered_storage_tests {
    use super::*;

    #[test]
    fn test_alloc_value_with_offset_basic() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        let (offset1, val1) = arena.alloc_value_with_offset(42u64).unwrap();
        let (offset2, val2) = arena.alloc_value_with_offset(100u64).unwrap();

        // First allocation should be at offset 0 (aligned)
        assert_eq!(offset1, 0);
        // Second allocation should be after the first
        assert!(offset2 > offset1);
        assert!(offset2 >= std::mem::size_of::<u64>() as u64);

        // Values should be correct
        assert_eq!(*val1, 42);
        assert_eq!(*val2, 100);

        // Mutation should work
        *val1 = 999;
        assert_eq!(*val1, 999);
    }

    #[test]
    fn test_read_at_basic() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        let (offset, _) = arena.alloc_value_with_offset(12345u64).unwrap();

        // Read it back
        // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
        let value: &u64 = unsafe { arena.read_at(offset) };
        assert_eq!(*value, 12345);
    }

    #[test]
    fn test_read_at_mut_basic() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        let (offset, _) = arena.alloc_value_with_offset(42u64).unwrap();

        // Read and modify
        // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
        let value: &mut u64 = unsafe { arena.read_at_mut(offset) };
        assert_eq!(*value, 42);
        *value = 100;

        // Verify modification persisted
        // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
        let value: &u64 = unsafe { arena.read_at(offset) };
        assert_eq!(*value, 100);
    }

    #[test]
    fn test_alloc_value_with_offset_struct() {
        #[derive(Debug, Clone, PartialEq)]
        struct TestNode {
            id: u64,
            name: [u8; 32],
            value: i32,
        }

        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        let node = TestNode {
            id: 12345,
            name: [b'A'; 32],
            value: -999,
        };

        let (offset, stored) = arena.alloc_value_with_offset(node.clone()).unwrap();
        assert_eq!(stored.id, 12345);
        assert_eq!(stored.value, -999);

        // Read it back
        // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
        let read: &TestNode = unsafe { arena.read_at(offset) };
        assert_eq!(read.id, node.id);
        assert_eq!(read.name, node.name);
        assert_eq!(read.value, node.value);
    }

    #[test]
    fn test_alloc_value_with_offset_alignment() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        // Allocate a byte first to potentially misalign
        let (offset1, _) = arena.alloc_value_with_offset(1u8).unwrap();
        assert_eq!(offset1, 0);

        // Now allocate a u64 which requires 8-byte alignment
        let (offset2, val) = arena.alloc_value_with_offset(42u64).unwrap();

        // offset2 should be 8-byte aligned
        assert_eq!(offset2 % 8, 0);
        assert_eq!(*val, 42);
    }

    #[test]
    fn test_alloc_value_with_offset_multiple() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        let mut offsets = Vec::new();
        for i in 0..100u64 {
            let (offset, val) = arena.alloc_value_with_offset(i).unwrap();
            offsets.push(offset);
            assert_eq!(*val, i);
        }

        // All offsets should be unique and in ascending order
        for window in offsets.windows(2) {
            assert!(window[0] < window[1]);
        }

        // Read all values back
        for (i, offset) in offsets.iter().enumerate() {
            // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
            let val: &u64 = unsafe { arena.read_at(*offset) };
            assert_eq!(*val, i as u64);
        }
    }

    #[test]
    fn test_arena_allocator_with_offset() {
        let allocator = ArenaAllocator::with_chunk_size(4096).unwrap();

        let epoch = allocator.current_epoch();
        let arena = allocator.arena(epoch).unwrap();

        let (offset, val) = arena.alloc_value_with_offset(42u64).unwrap();
        assert_eq!(*val, 42);

        // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
        let read: &u64 = unsafe { arena.read_at(offset) };
        assert_eq!(*read, 42);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "exceeds chunk used bytes")]
    fn test_read_at_out_of_bounds() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();
        let (_offset, _) = arena.alloc_value_with_offset(42u64).unwrap();

        // Read way past the allocated region: should panic in debug
        // SAFETY: intentionally invalid offset to test debug assertion
        unsafe {
            let _: &u64 = arena.read_at(4000);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "is not aligned")]
    fn test_read_at_misaligned() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();
        // Allocate a u8 at offset 0
        let (_offset, _) = arena.alloc_value_with_offset(0xFFu8).unwrap();
        // Also allocate some bytes so offset 1 is within used range
        let _ = arena.alloc_value_with_offset(0u64).unwrap();

        // Try to read a u64 at offset 1 (misaligned for u64)
        // SAFETY: intentionally misaligned offset to test debug assertion
        unsafe {
            let _: &u64 = arena.read_at(1);
        }
    }

    #[test]
    #[cfg(not(miri))] // parking_lot uses integer-to-pointer casts incompatible with Miri strict provenance
    fn test_concurrent_read_stress() {
        use std::sync::Arc;

        let arena = Arc::new(Arena::with_chunk_size(EpochId::INITIAL, 1024 * 1024).unwrap());
        let num_threads = 8;
        let values_per_thread = 1000;

        // Each thread allocates values and records offsets
        let mut all_offsets = Vec::new();
        for t in 0..num_threads {
            let base = (t * values_per_thread) as u64;
            let mut offsets = Vec::with_capacity(values_per_thread);
            for i in 0..values_per_thread as u64 {
                let (offset, _) = arena.alloc_value_with_offset(base + i).unwrap();
                offsets.push(offset);
            }
            all_offsets.push(offsets);
        }

        // Now read all values back concurrently from multiple threads
        let mut handles = Vec::new();
        for (t, offsets) in all_offsets.into_iter().enumerate() {
            let arena = Arc::clone(&arena);
            let base = (t * values_per_thread) as u64;
            handles.push(std::thread::spawn(move || {
                for (i, offset) in offsets.iter().enumerate() {
                    // SAFETY: offset was returned by alloc_value_with_offset for the same type and arena
                    let val: &u64 = unsafe { arena.read_at(*offset) };
                    assert_eq!(*val, base + i as u64);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_alloc_value_with_offset_grows_chunks() {
        // Create a tiny arena where a single chunk won't fit everything
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 64).unwrap();

        // Fill up the first chunk
        let (offset1, _) = arena.alloc_value_with_offset([0u8; 48]).unwrap();
        // Chunk 0, low offset
        assert_eq!(offset1 >> 48, 0); // chunk index = 0

        // This overflows chunk 0 → should transparently allocate chunk 1
        let (offset2, val2) = arena.alloc_value_with_offset([42u8; 32]).unwrap();
        assert_eq!(offset2 >> 48, 1); // chunk index = 1
        assert_eq!(*val2, [42u8; 32]);

        // Read back from chunk 1
        let read_back: &[u8; 32] = unsafe { arena.read_at(offset2) };
        assert_eq!(*read_back, [42u8; 32]);

        // Arena should now have 2 chunks
        assert_eq!(arena.stats().chunk_count, 2);
    }

    #[test]
    fn test_multi_type_interleaved() {
        #[derive(Debug, Clone, PartialEq)]
        #[repr(C)]
        struct Record {
            id: u64,
            flags: u32,
            weight: f32,
        }

        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096).unwrap();

        // Interleave different types
        let (off_u8, _) = arena.alloc_value_with_offset(0xAAu8).unwrap();
        let (off_u32, _) = arena.alloc_value_with_offset(0xBBBBu32).unwrap();
        let (off_u64, _) = arena.alloc_value_with_offset(0xCCCCCCCCu64).unwrap();
        let (off_rec, _) = arena
            .alloc_value_with_offset(Record {
                id: 42,
                flags: 0xFF,
                weight: std::f32::consts::PI,
            })
            .unwrap();

        // Read them all back
        // SAFETY: all offsets were returned by alloc_value_with_offset for matching types and arena
        unsafe {
            assert_eq!(*arena.read_at::<u8>(off_u8), 0xAA);
            assert_eq!(*arena.read_at::<u32>(off_u32), 0xBBBB);
            assert_eq!(*arena.read_at::<u64>(off_u64), 0xCCCCCCCC);

            let rec: &Record = arena.read_at(off_rec);
            assert_eq!(rec.id, 42);
            assert_eq!(rec.flags, 0xFF);
            assert!((rec.weight - std::f32::consts::PI).abs() < 0.001);
        }
    }
}
