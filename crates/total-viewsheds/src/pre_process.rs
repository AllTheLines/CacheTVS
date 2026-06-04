//! Code to run before the main computations.

use std::collections::HashMap;

use color_eyre::eyre::{ContextCompat as _, Result};

/// Create a fast lookup for deciding which points should have their viewsheds saved.
pub fn create_biggest_tvs_subgrid(config: &crate::config::Compute) -> Result<HashMap<i64, i64>> {
    let neighbourhood_size = config
        .only_save_biggest_viewsheds
        .context("Must specify --only-save-biggest-viewsheds")?;
    let tvs_source_path = config
        .tvs_source_path
        .clone()
        .context("Must specify --tvs-source-path")?;
    let dataset = gdal::Dataset::open(&tvs_source_path)?;
    let hashmap = find_biggest_total_surfaces(&dataset, i64::from(neighbourhood_size))?;

    Ok(hashmap)
}

/// Find the biggest surface areas within a subgrid of a computed TVS heatmap .tiff.
pub fn find_biggest_total_surfaces(
    dataset: &gdal::Dataset,
    neighbourhood_size: i64,
) -> Result<HashMap<i64, i64>> {
    let mut hashmap = HashMap::new();
    let mut contenders = HashMap::new();
    let size = dataset.raster_size();
    let band = dataset.rasterband(1)?;
    let points = band.read_as::<f32>((0, 0), size, size, None)?;

    let width = i64::try_from(points.len().isqrt())?;
    for (index_usize, point) in points.into_iter().enumerate() {
        let dem_id = tvs_id_to_dem_id(i64::try_from(index_usize)?, width);
        let neigbourhood_id = get_neighbourhood_id(dem_id, width * 3, neighbourhood_size);
        if let Some(contender) = contenders.get_mut(&neigbourhood_id) {
            if &point > contender {
                *contender = point;
                hashmap
                    .entry(neigbourhood_id)
                    .and_modify(|entry| *entry = dem_id);
            }
        } else {
            contenders.insert(neigbourhood_id, point);
            hashmap.insert(neigbourhood_id, dem_id);
        }
    }

    Ok(hashmap)
}

/// Find which neighbourhood a given index is in.
pub const fn get_neighbourhood_id(index: i64, global_width: i64, neighbourhood_size: i64) -> i64 {
    let global_x = index.rem_euclid(global_width);
    let global_y = index.div_euclid(global_width);

    let neighbourhood_width = neighbourhood_size.isqrt();
    let neighbourhoods_per_row = global_width.div_euclid(neighbourhood_width);
    let neighbourhood_x = global_x.div_euclid(neighbourhood_width);
    let neighbourhood_y = global_y.div_euclid(neighbourhood_width);
    (neighbourhood_y * neighbourhoods_per_row) + neighbourhood_x
}

/// Convert a TVS ID to a DEM ID.
const fn tvs_id_to_dem_id(tvs_id: i64, width: i64) -> i64 {
    let tvs_x = tvs_id.rem_euclid(width);
    let tvs_y = tvs_id.div_euclid(width);
    let dem_x = tvs_x + width;
    let dem_y = tvs_y + width;
    let dem_width = width * 3;
    (dem_y * dem_width) + dem_x
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn tvs_neighbourhood_id() {
        fn run(index: i64) -> i64 {
            let global_width = 20;
            let neighbourhood_size = 16;
            get_neighbourhood_id(index, global_width, neighbourhood_size)
        }

        assert_eq!(run(0), 0);
        assert_eq!(run(3), 0);
        assert_eq!(run(60), 0);
        assert_eq!(run(63), 0);

        assert_eq!(run(4), 1);
        assert_eq!(run(44), 1);

        assert_eq!(run(19), 4);
        assert_eq!(run(79), 4);

        assert_eq!(run(80), 5);
        assert_eq!(run(83), 5);

        assert_eq!(run(383), 20);

        assert_eq!(run(399), 24);
        assert_eq!(run(400), 25);
    }

    #[test]
    fn biggest_total_surfaces_subgrid() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "Just tests, these aren't big values"
        )]
        let data = (0..400).map(|i| i as f32).collect();
        let dataset = crate::tests::fixtures::create_tvs_tiff(data).unwrap();
        let hashmap = find_biggest_total_surfaces(&dataset, 16).unwrap();

        // Top left
        assert!(!hashmap.contains_key(&79));
        assert_eq!(hashmap[&80], 1403);
        assert_eq!(hashmap[&81], 1407);

        // Bottom right
        assert_eq!(hashmap[&143], 2375);
        assert_eq!(hashmap[&144], 2379);
        assert!(!hashmap.contains_key(&145));
    }
}
