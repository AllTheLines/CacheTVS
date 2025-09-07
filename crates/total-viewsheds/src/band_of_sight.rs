//! The Band of Sight (`BoS`) is essentially the line of sight. In other
//! words: which DEM points fall along a line 'as the crow flies'.
//!
//! The conception of the `BoS` here is different to traditional 'V'-shaped
//! methods. Here we use a parallel band. Consider the following DEM,
//! where '/' represents the sides of a single `BoS`:
//!
//!    .  .  .  .  +  /  .  .  .
//!    .  .  .  /  +  .  .  .  .
//!    .  .  .  +  /  .  .  .  .
//!    .  .  /  +  .  .  .  .  .
//!    .  . (+) /  .  .  .  .  .
//!    .  /  +  .  .  .  .  .  .
//!    .  +  /  .  .  .  .  .  .
//!    /  +  .  .  .  .  .  .  .
//!    +  /  .  .  .  .  .  .  .
//!
//! The '(+)' is the viewer, or the Point of View (`PoV`) as it's referred
//! to here. The other '+'s are the points within the band of sight. In
//! effect there are actually 2 bands here, one from looking forward (the
//! front band) and another looking backwards (the back band).
//!
//! Even though `BoS`s are needed for every point (well every point that
//! doesn't have its line of sight truncated by the edge of the DEM) we
//! only need to calculate the *shape* of one `BoS` per sector angle. That
//! shape can then be applied to every point. The shape is described by
//! deltas describing the relative path from any given point. As far as I
//! know describing `BoS` shapes is a novel contribution to the state of the
//! art in viewshed analysis - I'd be very interested to know if you know
//! of any previous precedent.
//!
//! To create a single `BoS` from which to define the master shape for a
//! sector, we use the sorted DEM points from the `Axes` struct. Using the
//! `sector_ordered` points we can populate the `BoS` but we don't know the
//! perpendicular distance each '+' point is from the `PoV` '(+)'. To then
//! order this specific subset of the DEM we use the `sight_ordered`
//! points.
//!
//! Note that the size of the `BoS` is defined via the maximum line of sight.
//! The maximum line of sight is converted to the number of non-diagonal
//! DEM points that would reach such a distance and that is used to define the
//! size of the `BoS`. Note that for diagonal distances the same number of points
//! will actually reach further, namely √2 times further in the extreme case,
//! (calculated from how much longer the diagonal of a square is than its width).
//! So this over-fitting shouldn't be a problem.

use color_eyre::Result;

/// How much bigger the diagonal of a square is compared to its width.
const DIAGANOL_GREATER_THAN_WIDTH: f64 = core::f64::consts::SQRT_2;

impl crate::dem::DEM {
    /// The number of deltas in a single band.
    pub const fn band_deltas_size(&self) -> u32 {
        // We subtract one because there are always 1 fewer deltas in a set than there are actual
        // items in the set.
        self.band_size - 1
    }

    /// These deltas apply to *all* possible bands in this sector.
    pub fn compile_band_data(&mut self) -> Result<()> {
        tracing::debug!("Calculating band deltas for {}°", self.axes.angle);

        let mut dem_ids_to_compute = Vec::new();
        let mut distance_ids = Vec::new();

        self.band_deltas = vec![0i32; usize::try_from(self.band_deltas_size())?];
        self.band_distances = vec![0f32; usize::try_from(self.band_deltas_size())?];

        // In order for the line of sight not to "slip through" DEM points because it's too thin,
        // we need to ensure a minimum density of points. I don't think this step is mentioned in
        // the original paper, so I could be wrong. My thinking is that light cannot slip through
        // a perfect horizontal or vertical line of points. So the density in such a case is
        // exactly one point per point length. But in the diagonal worst case, say 45°, that same
        // density is not naturally reached. Another way to look at this is imagining that the band
        // of sight is created from 2 parallel lines of consistent distance from each other. In say
        // the horizontal case the band is naturally filled with 1 point for every point length,
        // but in the diagonal case it's only filled with 1/√2 points per point length. So in order
        // to handle the worst case we must increase the density by √2.
        //
        // Note that this doesn't actually increase the size of the final band deltas vector. These
        // are just the samples from which we calculate the deltas.
        //
        // TODO:
        // Should this be calculated on a per-angle basis? So that each band has the same affective
        // density of points?
        #[expect(
            clippy::as_conversions,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "
              Accuracy is not required here, we just need a big enough numnber for the worst case.
            "
        )]
        let band_samples = (f64::from(self.band_size * 2) * DIAGANOL_GREATER_THAN_WIDTH) as usize;

        // Find the very middle of the DEM. This is arbitrary as we ultimately only care about
        // deltas.
        let middle_of_array = usize::try_from(self.size.div_euclid(2))?;

        // We can't just start from the point of view and work towards the end of a band. The
        // sector ordered DEM IDs are ordered from their distance from the line-like _spine_ of
        // the band of sight. There's no guaranteed way of finding the point-like _centre_ of a
        // band of sight without also querying the sight-ordered array as well. So here we just
        // populate a band with "random" DEM IDs from around the middle of the sector-ordered
        // grid
        let band_edge = middle_of_array - band_samples.div_euclid(2);
        #[expect(
            clippy::indexing_slicing,
            reason = "It's easier to read and panicking is appropriate as there's no way to recover"
        )]
        for i in 0..band_samples {
            let next_band_point = band_edge + i;
            let dem_id = self.axes.sector_ordered[next_band_point];
            dem_ids_to_compute.push(dem_id);

            // Make a note of the distance IDs (how far each point is from a line orthogonal to
            // the sector ordering).
            let distance_id = self.axes.sight_ordered_map[usize::try_from(dem_id)?];
            distance_ids.push(distance_id);
        }

        // These IDs map the distance IDs to the DEM IDs. Let's say that the DEM IDs we need to
        // compute are:
        //   `31, 22, 46`
        // Then we look up those IDs on the sight-ordered map and we get another set of IDs.
        // Note that there is no direct intuition for what these IDs are, they just represent how
        // the DEM IDs are ordered based on another set of axes. So let's just say they are:
        //   `83, 91, 57`
        // Now because this second set is connected to the first set, any operation that we do on
        // the second can be equally applied to the first. The smallest ID in the second set is
        // `57` which is in the third position, etc. And so to map the entire second set from
        // smallest to largest is:
        //   `2, 0, 1`
        // That map can then be used to order the first set. That is what we're doing here.
        let mut distances_to_sector_ids_map: Vec<usize> = (0..band_samples).collect();
        #[expect(
            clippy::indexing_slicing,
            reason = "`distances_to_sector_ids_map` is always smaller than `distance_ids`"
        )]
        distances_to_sector_ids_map.sort_by(|&i, &j| {
            let left = distance_ids[i];
            let right = distance_ids[j];
            left.partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fill the actual band of sight with the deltas that trace a path from the PoV
        // to the final point in the band.
        let pov_in_band = band_samples.div_euclid(2);
        let end_of_band = pov_in_band + usize::try_from(self.band_deltas_size())?;

        #[expect(
            clippy::indexing_slicing,
            reason = "It's easier to read and panicking is appropriate as there's no way to recover"
        )]
        {
            let pov_sector_id = distances_to_sector_ids_map[pov_in_band];
            let pov_dem_id = usize::try_from(dem_ids_to_compute[pov_sector_id])?;
            let pov_distance = self.axes.distances[pov_dem_id];
            for (i, band_id) in (pov_in_band..end_of_band).enumerate() {
                let current_sector_id = distances_to_sector_ids_map[band_id];
                let current_dem_id = i32::try_from(dem_ids_to_compute[current_sector_id])?;
                let next_dem_id =
                    i32::try_from(dem_ids_to_compute[distances_to_sector_ids_map[band_id + 1]])?;

                let delta = current_dem_id - next_dem_id;
                self.band_deltas[i] = delta;

                let next_distance = self.axes.distances[usize::try_from(next_dem_id)?];
                self.band_distances[i] = (pov_distance - next_distance).abs();
            }
        }

        Ok(())
    }
}

// This is a little map to help orient yourself when looking at the test results:
//
// 0  1  2  3  4  5  6  7  8
// 9  10 11 12 13 14 15 16 17
// 18 19 20 21 22 23 24 25 26
// 27 28 29 30 31 32 33 34 35
// 36 37 38 39 40 41 42 43 44
// 45 46 47 48 49 50 51 52 53
// 54 55 56 57 58 59 60 61 62
// 63 64 65 66 67 68 69 70 71
// 72 73 74 75 76 77 78 79 80
#[expect(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::unreadable_literal,
    clippy::approx_constant,
    reason = "Test input is known"
)]
#[cfg(test)]
mod test {
    use googletest::prelude::*;

    /// Reconstruct bands from deltas.
    fn reconstruct_bands(angle: u16) -> (Vec<Vec<u32>>, Vec<f32>) {
        let mut dem = crate::dem::DEM::new(
            crate::projection::LatLonCoord(geo::Coord::zero()),
            9,
            1.0,
            3,
        )
        .unwrap();
        assert_eq!(dem.computable_points_count, 9);
        dem.calculate_axes(f32::from(angle)).unwrap();

        let mut both_bands = Vec::new();
        dem.compile_band_data().unwrap();

        for point in 0..dem.computable_points_count {
            let mut front_band = Vec::new();
            let mut back_band = Vec::new();
            let mut front_dem_id = dem.tvs_id_to_pov_id(point);
            let mut back_dem_id = front_dem_id;
            front_band.push(front_dem_id);
            back_band.push(back_dem_id);
            for delta in &dem.band_deltas {
                front_dem_id = (front_dem_id as i32 + delta) as u32;
                back_dem_id = (back_dem_id as i32 - delta) as u32;
                front_band.push(front_dem_id);
                back_band.push(back_dem_id);
            }

            // The end of the back band should be the same as the start as the front band. So
            // it makes sense to push the back band before the front one.
            back_band.reverse();
            both_bands.push(back_band.clone());
            both_bands.push(front_band.clone());
        }

        (both_bands, dem.band_distances)
    }

    #[test]
    fn sector_at_0_degrees() {
        let (bands, distances) = reconstruct_bands(0);
        #[rustfmt::skip]
        assert_eq!(
            bands,
            [
                [57, 48, 39, 30], [30, 21, 12, 3],
                [58, 49, 40, 31], [31, 22, 13, 4],
                [59, 50, 41, 32], [32, 23, 14, 5],
                [66, 57, 48, 39], [39, 30, 21, 12],
                [67, 58, 49, 40], [40, 31, 22, 13],
                [68, 59, 50, 41], [41, 32, 23, 14],
                [75, 66, 57, 48], [48, 39, 30, 21],
                [76, 67, 58, 49], [49, 40, 31, 22],
                [77, 68, 59, 50], [50, 41, 32, 23]
            ]
        );

        assert_eq!(distances, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn sector_at_15_degrees() {
        let (bands, distances) = reconstruct_bands(15);
        #[rustfmt::skip]
        assert_eq!(
            bands,
            [
                [49, 48, 39, 30], [30, 21, 12, 11],
                [50, 49, 40, 31], [31, 22, 13, 12],
                [51, 50, 41, 32], [32, 23, 14, 13],
                [58, 57, 48, 39], [39, 30, 21, 20],
                [59, 58, 49, 40], [40, 31, 22, 21],
                [60, 59, 50, 41], [41, 32, 23, 22],
                [67, 66, 57, 48], [48, 39, 30, 29],
                [68, 67, 58, 49], [49, 40, 31, 30],
                [69, 68, 59, 50], [50, 41, 32, 31]]
        );

        assert_eq!(distances, [0.9659214, 1.9318428, 2.1906786]);
    }

    #[test]
    fn sector_at_90_degrees() {
        let (bands, distances) = reconstruct_bands(90);
        #[rustfmt::skip]
        assert_eq!(
            bands,
            [
                [33, 32, 31, 30], [30, 29, 28, 27],
                [34, 33, 32, 31], [31, 30, 29, 28],
                [35, 34, 33, 32], [32, 31, 30, 29],
                [42, 41, 40, 39], [39, 38, 37, 36],
                [43, 42, 41, 40], [40, 39, 38, 37],
                [44, 43, 42, 41], [41, 40, 39, 38],
                [51, 50, 49, 48], [48, 47, 46, 45],
                [52, 51, 50, 49], [49, 48, 47, 46],
                [53, 52, 51, 50], [50, 49, 48, 47]
            ]
        );

        assert_eq!(distances, [1.0000002, 2.0000002, 3.0000002]);
    }

    #[test]
    fn sector_at_135_degrees() {
        let (bands, distances) = reconstruct_bands(135);
        #[rustfmt::skip]
        assert_eq!(
            bands,
            [
                [6,  14, 22, 30], [30, 38, 46, 54],
                [7,  15, 23, 31], [31, 39, 47, 55],
                [8,  16, 24, 32], [32, 40, 48, 56],
                [15, 23, 31, 39], [39, 47, 55, 63],
                [16, 24, 32, 40], [40, 48, 56, 64],
                [17, 25, 33, 41], [41, 49, 57, 65],
                [24, 32, 40, 48], [48, 56, 64, 72],
                [25, 33, 41, 49], [49, 57, 65, 73],
                [26, 34, 42, 50], [50, 58, 66, 74]
            ]
        );

        assert_eq!(distances, [1.4142135, 2.828427, 4.242641]);
    }

    #[gtest]
    fn longer_bands_should_not_have_empty_regions() {
        let mut dem = crate::dem::DEM::new(
            crate::projection::LatLonCoord(geo::Coord::zero()),
            100,
            1.0,
            33,
        )
        .unwrap();
        dem.calculate_axes(0.0).unwrap();
        dem.compile_band_data().unwrap();
        expect_ne!(dem.band_deltas.last().unwrap(), &0i32);
        expect_ne!(dem.band_distances.last().unwrap(), &0.0);
    }
}
