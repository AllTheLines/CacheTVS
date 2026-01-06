/// los contains all the traits necessary for implementing a line of sight algorithm
mod los;

mod rotation;

/// kernel is the exported kernel module
pub mod kernel;
mod unrolled_los;
mod vector_intrinsics;

pub use kernel::kernel;
