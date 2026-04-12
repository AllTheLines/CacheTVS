use crate::cpu::los::LineOfSight as _;
use crate::cpu::unrolled_los::UnrolledVectorLos;
use crate::cpu::vector_intrinsics::DEFAULT_VECTOR_LENGTH;
use crate::los_pack::LineOfSightPacked;
use itertools::izip;
use crate::cpu::rotation::lines;

/// The data output by a single angle.
pub struct OutputData {
    /// The visibile surface area.
    pub surfaces: Vec<f32>,
    /// The longest lines of sight.
    pub longest: Vec<LineOfSightPacked>,
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "so long as max_los < 2^24, the following as conversions are entirely safe"
)]
#[expect(
    clippy::integer_division,
    reason = "i32 is constructed from (i32, i32) converting back should succeed"
)]
/// `dem_to_pov` turns the `dem_id` to the `pov_id` so that the result can be stored in a heatmap
fn dem_id_to_tvs_id(dem_id: i64, width: usize, max_los: usize) -> i64 {
    let dem_x = (dem_id / width as i64) - max_los as i64;
    let dem_y = (dem_id % width as i64) - max_los as i64;

    let radius = (max_los - 1) as f32 / 2.0;
    let circ_x = dem_x as f32 - radius;
    let circ_y = dem_y as f32 - radius;

    let dist = circ_x.hypot(circ_y);
    if dist < radius {
        dem_x * (max_los as i64) + dem_y
    } else {
        -1
    }
}

/// `DEFAULT_UNROLL` is the default loop unrolling constant, which is based
/// off of the default vector length. 8-way unrolling for both the 4 and 8 wide
/// vectors, and 10-way unrolling for the 16-wide vector as it is optimal for Turins
const DEFAULT_UNROLL: usize = const {
    match DEFAULT_VECTOR_LENGTH {
        4 | 8 | 16 => 10,
        #[expect(
            clippy::unreachable,
            reason = "no one should be setting any other constants"
        )]
        _ => unreachable!(),
    }
};



/// `kernel` will calculate the longest line of sight heatmap for a given angle and elevation map
/// assuming that the maximum line of sight is `max_los`
pub fn kernel(
    db_worker: &super::storage::worker::Worker,
    elevation_map: &[i16],
    output: &mut OutputData,
    max_los: usize,
    angle: f32,
    config: &crate::run::compute::Config,
) {
    let surfaces = &mut output.surfaces;
    let longest = &mut output.longest;

    assert_eq!(
         surfaces.len(),
         max_los * max_los,
         "surfaces should be max_los squared length"
    );

    assert_eq!(
        longest.len(),
        max_los*max_los,
        "longest lines should be max_los squared length"
    );

    let mut vs =
        UnrolledVectorLos::<DEFAULT_UNROLL, DEFAULT_VECTOR_LENGTH>::new(max_los, config.refraction, config.scale);

    for (line, line_indexes) in lines(elevation_map, max_los, angle.into()) {
        for (pov, (&pov_height, &result_dem_id)) in
            izip!(line.iter().take(max_los), line_indexes.iter().take(max_los)).enumerate()
        {
            let result_tvs_id = dem_id_to_tvs_id(result_dem_id, 3 * max_los, max_los);

            // if the line of sight is not within our computable points, do not consider it
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_wrap,
                reason = "max_los^2 < 2^31"
            )]
            if result_tvs_id < 0i64 || result_tvs_id >= (max_los * max_los) as i64 {
                continue;
            }

            let neighbor = pov + 1;

            #[expect(
                clippy::indexing_slicing,
                reason = "if slicing is out of bounds, it should panic"
            )]
            let (point_surface, point_longest, point_visibility) = vs.line_of_sight(
                f32::from(pov_height) + config.observer_height,
                &line[neighbor..neighbor + max_los],
            );

            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "max_los^2 < 2^31"
            )]
            {
                // safety: result_tvs_id is guaranteed to be within [0..max_los^2)
                unsafe {
                    *surfaces.get_unchecked_mut(result_tvs_id as usize) = point_surface;
                };
                // safety: result_tvs_id is guaranteed to be within [0..max_los^2)
                unsafe {
                    *longest.get_unchecked_mut(result_tvs_id as usize) = LineOfSightPacked::new_unchecked(point_longest as u32, angle as u16)
                };
                if cfg!(any(test, feature = "ring_data"))
                    && crate::run::compute::Compute::is_process_viewsheds(&config.process)
                {
                    db_worker.store_bitmap(result_dem_id as u32, angle as u16, &point_visibility);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{cpu::kernel as cpu_kernel, los_pack::LineOfSightPacked, run::compute::test::default_config};
    use crate::cpu::kernel::OutputData;

    #[test]
    fn total_surfaces() {
        let dem = &kernel::tests::dems::bigger_dem();
        let db_worker = crate::cpu::storage::worker::Worker::new_noop();
        let config = default_config(
            crate::config::Backend::CPU,
            &tempfile::NamedTempFile::new().unwrap(),
        );

        let mut forward = OutputData {
            surfaces: vec![0.0f32; 4*4],
            longest: vec![LineOfSightPacked::new(0, 0).unwrap(); 4 * 4]
        };

        cpu_kernel(&db_worker, dem, &mut forward, 4, 0.0, &config);

        let mut backward = OutputData {
            surfaces: vec![0.0f32; 4*4],
            longest: vec![LineOfSightPacked::new(0, 0).unwrap(); 4 * 4]
        };

        cpu_kernel(&db_worker, dem, &mut backward, 4, 180.0, &config);

        let result = forward
            .surfaces
            .iter()
            .zip(backward.surfaces.iter())
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(
            result,
            [
                0.0, 0.0,        0.0,       0.0,
                0.0, 0.0349066,  0.1570797, 0.0,
                0.0, 0.19198631, 0.34906602,0.0,
                0.0, 0.0,        0.0,       0.0,
            ]
        );
    }
}
