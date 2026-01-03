/// los contains all the traits necessary for implementing a line of sight algorithm
mod los;

mod rotation;

/// vector contains vectorized implementations of the line of sight traits
mod vector;

/// kernel is the exported kernel module
pub mod kernel;
pub use kernel::kernel;
