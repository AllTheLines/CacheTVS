#![expect(
    // TODO: use `get_unchecked()` for a potential speed up?
    clippy::indexing_slicing,
    reason = "This needs to be able to run on the GPU"
)]
// You can use this for debugging when running with `--compute cpu`
// #[cfg(not(target_arch = "spirv"))]
// dbg!(distance);

//! Viewshed kernel. The heart of the calculations.

/// Ensure that the first point from the point of view is always visible
const MAX_ANGLE: f32 = -2000.0;

/// Normalisation constant: due to repeated exposure of closer points and
/// infrequent exposure of points further away. See main loop for more explanation.
const TAN_ONE_RAD: f32 = 0.017_453_3;

/// So that some points are not visible simply by virtue of the earth's spherical
/// shape.
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

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
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
}

/// The direction of a band from the observer's point of view. Whether it points North or South is
/// not relevant, they're just opposite to each other.
enum BandDirection {
    /// A band facing forward from the observer.
    Forward,
    /// A band facing behind the observer.
    Backward,
}

/// The kernel
#[expect(clippy::too_many_lines, reason = "Will refactor soon")]
#[inline]
pub fn kernel(
    // The identifier that decides which PoV and band direction to calculate.
    // The current known longest line of sight is 538km. Let's say that the actual longest could reach
    // 600km. So for a DEM of 30m resolution, the maximum number of points we could be dealing with
    // is: `(500,000 / 30)^2 = 400,000,000`. The max of `u32` is 4,294,967,295.
    kernel_id: u32,
    // Constants for the calculations.
    constants: &Constants,
    // Every single DEM point's elevation.
    elevations: &[f32],
    // All the required distances for the band. All bands share the same values.
    distances: &[f32],
    // Deltas used to build a band. Deltas are always the same for both front and back bands.
    //
    // Deltas are simply the numerical difference between DEM IDs in a band of
    // sight. DEM IDs are certainly different for every point in a DEM, but
    // conveniently the _differences_ between DEM IDs are identical. With the only
    // caveat that back-facing bands have opposite magnitudes. It should be stressed
    // that this feature of band data is a huge benefit to the space-requirements
    // (and thus speed) of the algorithm.
    // TODO: Explore:
    //         * storing the band deltas in local memory.
    //         * compressing repeating delta values -- is simple addition faster than
    //           memory accesses?
    deltas: &[i32],
    // Array for final TVS values. Usually 1/8th the size of DEM.
    cumulative_surfaces: &mut [f32],
    // Array for keeping track of visible chunks called Ring Sectors.
    ring_data: &mut [u32],
) {
    let tvs_id: u32;
    let mut max_angle = MAX_ANGLE;
    let half_total_bands = constants.total_bands.div_euclid(2);
    let mut is_currently_visible = true;
    let mut is_previously_visible = true;
    let mut closing = false;

    // Keep track of the amount of the earth visible from this particular band
    let mut band_surface = 0.0;

    // For each visible continuous region we find, we'll note it as a single ring
    // sector. We need to know how many ring sectors we find in order to save that
    // data to file later. Start at 1 to reserve later writing of the number of ring
    // sectors found.
    let mut ring_id = 1;

    // Translate a kernel ID to a TVS ID
    let band_direction = if kernel_id < half_total_bands {
        tvs_id = kernel_id;
        BandDirection::Forward
    } else {
        tvs_id = kernel_id - half_total_bands;
        BandDirection::Backward
    };

    let pov_x = tvs_id.rem_euclid(constants.tvs_width) + constants.max_los_as_points;
    let pov_y = (tvs_id.div_euclid(constants.tvs_width)) + constants.max_los_as_points;
    #[expect(
        clippy::as_conversions,
        reason = "This needs to run on the GPU where fallibility isn't possible"
    )]
    let pov_id = ((pov_y * constants.dem_width) + pov_x) as usize;

    // The PoV is involved in every calculation, so do now and save for later.
    let pov_elevation = elevations[pov_id] + constants.observer_height;

    #[expect(
        clippy::as_conversions,
        reason = "This needs to run on the GPU where fallibility isn't possible"
    )]
    // This thread needs its own unshared space in global memory.
    let ring_data_start = (kernel_id * constants.reserved_rings_per_band) as usize;

    // We already know that the first point from the PoV is always visible and
    // is therefore the opening of a visible region.
    save_ring_data(
        ring_data,
        constants.reserved_rings_per_band,
        ring_data_start,
        ring_id,
        pov_id,
    );

    // The DEM ID will change as we loop, but the PoV won't. For now we need the PoV
    // ID to start the reconstruction of a unique band from the band delta template.
    let mut dem_id = pov_id;

    // The kernel's kernel. The most critical code of all.
    for band_index in 0..deltas.len() {
        let delta = deltas[band_index];

        // Derive the new DEM ID.
        dem_id = match band_direction {
            BandDirection::Forward => delta_add(dem_id, delta),
            BandDirection::Backward => delta_subtract(dem_id, delta),
        };

        // TODO: Is there a way to create the deltas so that they never create DEM IDs outside the
        // DEM? The issue is that bands of sight need to be different lengths for different angles.
        // Consider how the diagnols reach geographically further with the same number of points.
        if dem_id >= elevations.len() {
            break;
        }

        // Pull the actual data needed to make a visibility calculation from global memory.
        // TODO: does getting these all at once before the loop give a speed up?
        let elevation_delta = elevations[dem_id] - pov_elevation;
        let distance = distances[band_index];

        // The actual visibility calculation.
        // Note the adjustment for curvature of the earth. It is merely a crude
        // approximation using the spherical earth model.
        // TODO:
        //   * Account for refraction.
        //   * Is there a performance gain to be had from only checking for an
        //     increase in elevation as a trigger for the full angle calculation?
        //   * Is this safe for `f32`? At what point does it break down?
        let angle = (elevation_delta / distance) - (distance / EARTH_RADIUS_SQUARED);

        //                            5              |-
        //                        4 .-`-. 6          |-
        //            1       3  .-`     `-.  7      |-
        //   o    0 .-`-. 2   .-`           `-.      |- Elevation deltas
        //   /\  .-`     `-.-`                 `-.8  |-
        //    |                                      |-
        //    |---|---|---|---|---|---|---|---|---|
        //    |       Distance deltas
        //    |
        //   PoV  --------> direction of band point iterations
        //
        // Notice how only points 1, 4 and 5 increase the angle between the viewer and
        // the land and therefore can be considered visible.
        is_currently_visible = angle > max_angle;

        // Here we consider the *previous* visibility to decide whether this is the
        // beginning or ending of a visible region.
        let opening = is_currently_visible && !is_previously_visible;
        closing = is_previously_visible && !is_currently_visible;

        if is_currently_visible {
            // This normalises surfaces due to repeated visibility over sectors.
            // For example points near the PoV may be counted 180 times giving
            // an unrealistic aggregation of total surfaces. This trig operation
            // (0.017 is tan(1 degree) where 1 degree is the size of each sector)
            // considers adjacent points as 1/57th, points that are about 57 cells
            // away as 1 and points further away as increasingly over 1.
            // Bear in my mind that due to the parallel nature of the band of sight,
            // distant points may not even be swept for a given PoV, therefore it
            // is an approximation to over-compensate the surface area of very
            // distant points.
            //
            // Ultimately the reasoning for such normalisation is not so much to
            // measure true surface areas, but to ensure as much as possible that all
            // PoVs are treated consistently so that overlapping TVS raster map tiles
            // do not have visual artefacts.
            //
            // TODO: Can this be refactored into a single calculation at the closing
            //       of a ring sector?
            band_surface += distance * TAN_ONE_RAD;
        }

        // Store the position on the DEM where the visible region starts.
        if opening {
            ring_id += 1;
            save_ring_data(
                ring_data,
                constants.reserved_rings_per_band,
                ring_data_start,
                ring_id,
                dem_id,
            );
        }

        // Store the position on the DEM where the visible region ends.
        if closing {
            ring_id += 1;
            save_ring_data(
                ring_data,
                constants.reserved_rings_per_band,
                ring_data_start,
                ring_id,
                dem_id,
            );
        }

        // Prepare for the next iteration.
        is_previously_visible = is_currently_visible;
        max_angle = f32::max(angle, max_angle);
    }

    // Close any ring sectors prematurely cut off by a restricted line of sight.
    if is_currently_visible && !closing {
        ring_id += 1;
        save_ring_data(
            ring_data,
            constants.reserved_rings_per_band,
            ring_data_start,
            ring_id,
            dem_id,
        );
    }

    // Make a note at the start of the ring sector data of how many rings we found.
    save_ring_data(
        ring_data,
        constants.reserved_rings_per_band,
        ring_data_start,
        0,
        ring_id,
    );

    // Accumulate surfaces for a given DEM ID. Note that this is thread safe, because
    // front/back bands are kept in separate array positions. And that actual
    // cumulation additions are made on data from a *previous* run where a completely
    // different sector angle was calculated.
    #[expect(
        clippy::as_conversions,
        reason = "This needs to run on the GPU where fallibility isn't possible"
    )]
    {
        cumulative_surfaces[tvs_id as usize] += band_surface;
    }
}

/// Safely and efficiently add a `usize` and `i32`.
const fn delta_add(dem_id: usize, delta: i32) -> usize {
    #[expect(
        clippy::as_conversions,
        reason = "
          Our `usize`s are only ever created from `u32`s and deltas are calculated within
          the limits of `u32`.
        "
    )]
    let absolute = delta.unsigned_abs() as usize;

    if delta > 0 {
        dem_id + absolute
    } else {
        dem_id - absolute
    }
}

/// Safely and efficiently subtract a `i32` from a `usize`.
const fn delta_subtract(dem_id: usize, delta: i32) -> usize {
    #[expect(
        clippy::as_conversions,
        reason = "
          Our `usize`s are only ever created from `u32`s and deltas are calculated within
          the limits of `u32`.
        "
    )]
    let absolute = delta.unsigned_abs() as usize;

    if delta > 0 {
        dem_id - absolute
    } else {
        dem_id + absolute
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "
      `usize` values are only ever generated from `u32` values. So they can't truncate.
    "
)]
/// Save ring data.
fn save_ring_data(
    ring_data: &mut [u32],
    reserved_rings_per_band: u32,
    start: usize,
    index: usize,
    value: usize,
) {
    if index >= reserved_rings_per_band as usize {
        return;
    }
    ring_data[start + index] = value as u32;
}
