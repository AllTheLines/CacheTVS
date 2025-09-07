//! Manage computation cache.

use color_eyre::Result;

/// `Cache`
pub struct Cache {
    /// Total viewshed's own path on the file system.
    pub base_directory: std::path::PathBuf,
    /// The base directory for all the band data.
    band_deltas_directory: std::path::PathBuf,
    /// The base directory for all the distance data.
    distances_directory: std::path::PathBuf,
    /// The path to a band deltas cache, specific to both a DEM size and sector angle.
    band_deltas_path: std::path::PathBuf,
    /// The path to a distances cache, specific to both a DEM size and sector angle.
    distances_path: std::path::PathBuf,
    /// Did the base directories exists before we ensured they existed?
    pub is_cache_exists: bool,
}

impl Cache {
    /// Instantiate.
    pub fn new(state_directory: &std::path::Path, dem_width: u32, angle: u16) -> Self {
        let base_directory = state_directory
            .join("total-viewsheds")
            .join(format!("{dem_width}"));
        let band_deltas_directory = base_directory.join("band_deltas");
        let distances_directory = base_directory.join("distances");
        let band_deltas_path = band_deltas_directory.join(format!("{angle}.bin"));
        let distances_path = distances_directory.join(format!("{angle}.bin"));
        let is_cache_exists = band_deltas_path.exists() && distances_path.exists();

        Self {
            base_directory,
            band_deltas_directory,
            distances_directory,
            band_deltas_path,
            distances_path,
            is_cache_exists,
        }
    }

    /// Ensure that the required directories exist.
    pub fn ensure_directories_exists(&self) -> Result<()> {
        std::fs::create_dir_all(&self.band_deltas_directory)?;
        std::fs::create_dir_all(&self.distances_directory)?;
        Ok(())
    }

    /// Load band deltas from cache.
    pub fn load_band_deltas(&self) -> Result<Vec<i32>> {
        let delta_bytes = std::fs::read(&self.band_deltas_path)?;
        Ok(bytemuck::cast_slice(&delta_bytes).to_vec())
    }

    /// Load distances from cache.
    pub fn load_distances(&self) -> Result<Vec<f32>> {
        let distances = std::fs::read(&self.distances_path)?;
        Ok(bytemuck::cast_slice(&distances).to_vec())
    }

    /// Save band deltas to cache.
    pub fn save_band_deltas(&self, band_deltas: &[i32]) -> Result<()> {
        Ok(std::fs::write(
            &self.band_deltas_path,
            bytemuck::cast_slice(band_deltas),
        )?)
    }

    /// Save distances to cache.
    pub fn save_distances(&self, distances: &[f32]) -> Result<()> {
        Ok(std::fs::write(
            &self.distances_path,
            bytemuck::cast_slice(distances),
        )?)
    }
}

#[expect(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::indexing_slicing,
    reason = "They're just tests"
)]
#[cfg(test)]
mod test {
    #[test]
    fn saves_and_loads() {
        let state_directory = tempfile::tempdir().unwrap();
        let mut dem = crate::dem::DEM::new(
            crate::projection::LatLonCoord(geo::Coord::zero()),
            9,
            1.0,
            3,
        )
        .unwrap();
        let cache = crate::cache::Cache::new(state_directory.path(), dem.width, 45);
        cache.ensure_directories_exists().unwrap();
        dem.calculate_axes(35.0).unwrap();
        dem.compile_band_data().unwrap();
        cache.save_band_deltas(&dem.band_deltas).unwrap();
        cache.save_distances(&dem.axes.distances).unwrap();
        assert_eq!(
            cache.load_band_deltas().unwrap(),
            vec![-10i32, -9i32, -1i32]
        );
        let distances = cache.load_distances().unwrap();
        assert_eq!(distances.len(), 81);
        assert_eq!(distances[40], 5.570931);
        assert_eq!(distances.last().unwrap(), &11.141862);
    }
}
