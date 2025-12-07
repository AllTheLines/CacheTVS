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

pub mod chocolate_box;
pub mod constants;
pub mod elevations;
pub mod kernel;
mod ring_data;
pub mod rotation;

#[cfg(not(target_arch = "spirv"))]
/// Code used for tests and debugging.
pub mod tests {
    pub mod dems;
    pub mod matchers;
}

#[allow(
    clippy::allow_attributes,
    reason = "For some reason `expect` doesn't detect the veracity of the 'inline' lint"
)]
#[allow(
    clippy::missing_inline_in_public_items,
    clippy::too_many_arguments,
    reason = "SPIR-V requires an entrypoint"
)]
#[spirv(compute(threads(8, 8, 4)))]
/// The main entrypoint to the shader.
pub fn visibility(
    #[spirv(global_invocation_id)] id: glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] constants: &constants::Constants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elevations: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] cumulative_surfaces: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] ring_data: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] longest_lines: &mut [f32],
) {
    let linear_id = id.x
        + id.y * constants.dimensions.x
        + id.z * constants.dimensions.x * constants.dimensions.y;
    if linear_id >= constants.dimensions.w {
        return;
    }

    let mut buffers = kernel::Buffers {
        constants,
        elevations,
        cumulative_surfaces,
        longest_lines,
        ring_data,
    };

    kernel::Kernel::run(linear_id, &mut buffers);
}

#[allow(
    clippy::allow_attributes,
    reason = "For some reason `expect` doesn't detect the veracity of the 'inline' lint"
)]
#[allow(
    clippy::missing_inline_in_public_items,
    clippy::too_many_arguments,
    reason = "SPIR-V requires an entrypoint"
)]
#[spirv(compute(threads(8, 8, 4)))]
/// Entrypoint for rotating elevation data.
pub fn rotate(
    #[spirv(global_invocation_id)] id: glam::UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] constants: &constants::Constants,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] elevations_in: &[i16],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] elevations_out: &mut [f32],
) {
    let linear_id = id.x
        + id.y * constants.dimensions.x
        + id.z * constants.dimensions.x * constants.dimensions.y;
    if linear_id >= constants.dimensions.w {
        return;
    }

    let rotator = chocolate_box::Rotator::new_from_cached_trig(
        linear_id,
        constants.dem_width,
        constants.tvs_width,
        constants.sine,
        constants.cosine,
    );
    rotator.rotate_value_bilinear(elevations_in, elevations_out);
}
