

/// los contains all the traits necessary for implementing a line of sight algorithm
mod los;

mod rotation;

/// kernel is the exported kernel module
pub mod kernel;

/// `unrolled_los` holds a fully implemented los trait for unrolled vectorization
mod unrolled_los;

/// `vector_intrinsics` holds all the vector-related LOS intrinsics
mod vector_intrinsics;

pub use kernel::kernel;
