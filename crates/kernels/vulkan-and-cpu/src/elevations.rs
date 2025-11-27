//! Code for managing elevation data.

/// The nodata value from NASA's SRTM data.
pub const NODATA: f32 = -32768.0;

#[cfg_attr(not(target_arch = "spirv"), derive(Debug, PartialEq, Eq))]
#[expect(
    clippy::exhaustive_enums,
    reason = "There's never going to be more directions"
)]
/// The direction of a band from the observer's point of view. Whether it points North or South is
/// not relevant, they're just opposite to each other.
pub enum Direction {
    /// A line of sight facing forward from the observer.
    Forward,
    /// A line of sight facing behind the observer.
    Backward,
}

/// Manage fetching valid elevation data.
pub struct Elevations {
    /// Whether going forwards or backwards along the line of sight.
    pub direction: Direction,
    /// The DEM ID of the current eleveation point in rotated space.
    rotated_dem_id: usize,
    /// The elevation at the observer.
    pov_elevation: f32,
    /// A record of the last valid elevation. Used to fill "nodata" regions.
    ///
    /// Note that even though most, if not all, of these "nodata" regions occur at sea, that
    /// doesn't necessarily mean that a default elevation of 0 is best. There are various reasons
    /// why the sea isn't always at a perfect 0 elevation, which we won't go into here. The
    /// point being that we need to smoothly transition into "nodata" regions to avoid visual
    /// artefacts.
    last_valid_elevation: f32,
}

impl Elevations {
    #[inline]
    #[must_use]
    /// Instantiate.
    pub fn new(
        elevations: &[f32],
        direction: Direction,
        rotated_dem_id: usize,
        observer_height: f32,
    ) -> Self {
        let elevation = elevations[rotated_dem_id];
        let last_valid_elevation = if Self::is_valid(elevation) {
            elevation
        } else {
            0.0
        };
        let pov_elevation = last_valid_elevation + observer_height;

        Self {
            direction,
            rotated_dem_id,
            pov_elevation,
            last_valid_elevation,
        }
    }

    #[inline]
    /// Get the next elevation.
    pub fn next(&mut self, elevations: &[f32]) -> f32 {
        match self.direction {
            Direction::Forward => self.rotated_dem_id += 1,
            Direction::Backward => self.rotated_dem_id -= 1,
        }

        let elevation = elevations[self.rotated_dem_id];

        if Self::is_valid(elevation) {
            self.last_valid_elevation = elevation;
            elevation - self.pov_elevation
        } else {
            self.last_valid_elevation - self.pov_elevation
        }
    }

    /// Is the elevation valid.
    fn is_valid(elevation: f32) -> bool {
        let is_nodata = elevation <= NODATA;
        !elevation.is_nan() && !is_nodata
    }
}
