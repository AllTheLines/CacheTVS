//! Constants and flags for the kernel.

/// Constants that don't change for the entirety of the computation.
#[repr(C)]
#[cfg_attr(
    not(target_arch = "spirv"),
    derive(
        Copy, Clone, Default,
        // Bytemuck is what we use to cast data into raw bytes for CPU/GPU transfer.
        bytemuck::Zeroable, bytemuck::Pod,
    )
)]
#[expect(
    clippy::exhaustive_structs,
    clippy::pub_underscore_fields,
    reason = "We're only sharing this in the workspace"
)]
pub struct Constants {
    /// The number of invocations for each kernel dimensions. Needed to convert the dimensions into
    /// a scalar kernetl ID.
    pub dimensions: glam::UVec4,
    /// The total number of both forward and backward bands.
    pub total_bands: u32,
    /// The maximum distance that is expected to be possible. Units are the number of DEM points.
    pub max_los_as_points: u32,
    /// The original width of the DEM. Units are DEM points.
    pub dem_width: u32,
    /// The width of the computable region of the DEM. Units are DEM points.
    pub tvs_width: u32,
    /// The height of the observer in meters.
    pub observer_height: f32,
    /// The amount of memory reserved for storing computed ring sectors.
    pub reserved_rings_per_band: u32,
    /// Bitmask of what computations to process.
    pub process: u32,
    /// Padding.
    pub _pad0: u32,
}
impl Constants {
    #[inline]
    #[must_use]
    /// Should we be computing ring data?
    pub const fn is_ring_data(&self) -> bool {
        (self.process & (Flag::RingData.bit())) != 0
    }

    #[inline]
    #[must_use]
    /// Should we be computing ring data?
    pub const fn is_total_surfaces(&self) -> bool {
        (self.process & (Flag::TotalSurfaces.bit())) != 0
    }
}

#[repr(u32)]
#[expect(
    clippy::exhaustive_enums,
    reason = "We're only using it within our workspace"
)]
/// Bitmask of the computations to process.
pub enum Flag {
    /// Compute total surfaces.
    TotalSurfaces = 1 << 0,
    /// Compute ring data.
    RingData = 1 << 1,
}

impl Flag {
    #[expect(
        clippy::as_conversions,
        reason = "It's just a bit mask. Also, I don't know if there even is another way?"
    )]
    /// Just a single point to do cast from.
    #[inline]
    #[must_use]
    pub const fn bit(self) -> u32 {
        self as u32
    }
}
