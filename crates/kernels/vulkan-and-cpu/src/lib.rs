//! The most intensive code, aka the kernel. We keep it in a seperate crate so that it can be
//! compiled to shader representations.
//!
//! This crate can be run both on the GPU and CPU.

#![expect(
    // TODO: use `get_unchecked()` for a potential speed up?
    clippy::indexing_slicing,
    reason = "This needs to be able to run on the GPU"
)]
#![cfg_attr(target_arch = "spirv", no_std)]
#![expect(
    clippy::arithmetic_side_effects,
    reason = "`rust-gpu` is a subset of Rust and has some unique requirements"
)]

use spirv_std::spirv;

pub mod constants;
pub mod kernel;
mod ring_data;

/// The main entrypoint to the shader.
#[allow(
    clippy::allow_attributes,
    reason = "For some reason `expect` doesn't detect the veracity of the 'inline' lint"
)]
#[allow(
    clippy::missing_inline_in_public_items,
    reason = "SPIR-V requires an entrypoint"
)]
#[spirv(compute(threads(8, 8, 4)))]
pub fn main(
    #[spirv(global_invocation_id)] id: glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] constants: &constants::Constants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elevations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] distances: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] deltas: &[i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] cumulative_surfaces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] ring_data: &mut [u32],
) {
    let linear_id = id.x
        + id.y * constants.dimensions.x
        + id.z * constants.dimensions.x * constants.dimensions.y;
    if linear_id >= constants.dimensions.w {
        return;
    }

    kernel::kernel(
        linear_id,
        constants,
        elevations,
        distances,
        deltas,
        cumulative_surfaces,
        ring_data,
    );
}
