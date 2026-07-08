//! This is for joining polygons that wrap over the 360°-0° boundary. These joins are a bit more
//! complicated, but it's okay if the code is less efficient as it only happens once per viewshed
//! reconstruction.

use color_eyre::eyre::{ContextCompat as _, Result, bail};

use crate::vertices::Opening;

impl super::Joiner {
    /// The final angle faces the challenge of joining polygons together from the first angle.
    pub(crate) fn build_final_angle(&mut self) -> Result<()> {
        tracing::trace!("Building viewshed for final angle");

        let timing = std::time::Instant::now();
        self.active.sort_by_key(|polygon| polygon.furthest_opening);
        self.completed
            .sort_by_key(|polygon| polygon.furthest_opening);

        let starting_polygon_indexes: Vec<usize> = self
            .completed
            .iter()
            .enumerate()
            .filter(|item| item.1.is_created_at_angle_0)
            .map(|item| item.0)
            .collect();

        tracing::debug!("");
        let total_polygons = starting_polygon_indexes.len() + self.active.len();
        tracing::debug!("Final angle, polygons to check: {}", total_polygons);

        tracing::trace!(
            "Final angle, completed polygon (started at angle 0) indices: {:?}. Active polygons: {:?}",
            starting_polygon_indexes,
            self.active.len()
        );
        let mut touching_starting_polygons = Vec::new();
        for starting_polygon_index in starting_polygon_indexes {
            for final_polygon_index in 0..self.active.len() {
                if self
                    .handle_touching_polygons(final_polygon_index, Some(starting_polygon_index))?
                {
                    touching_starting_polygons.push(starting_polygon_index);
                }
            }
        }

        let polygons_to_remove = touching_starting_polygons.iter().rev();
        tracing::debug!(
            "Removing final joined polygons: {:?}",
            polygons_to_remove.clone().collect::<Vec<_>>()
        );
        for touching_starting_polygon in polygons_to_remove {
            self.completed.remove(*touching_starting_polygon);
        }

        let self_joining_polygon_indexes: Vec<usize> = self
            .active
            .iter()
            .enumerate()
            .filter(|item| item.1.is_created_at_angle_0)
            .map(|item| item.0)
            .collect();

        for self_joining_polygon_index in self_joining_polygon_indexes {
            self.handle_touching_polygons(self_joining_polygon_index, None)?;
        }

        tracing::debug!("Final angle done in {:?}", timing.elapsed());

        Ok(())
    }

    /// Check whether a final and a starting polygon are touching.
    fn handle_touching_polygons(
        &mut self,
        final_polygon_index: usize,
        maybe_starting_polygon_index: Option<usize>,
    ) -> Result<bool> {
        tracing::debug!(
            "Checking final polygon {final_polygon_index} \
             against starting polygon {maybe_starting_polygon_index:?}"
        );

        let vertices_clone = if let Some(starting_polygon_index) = maybe_starting_polygon_index {
            self.completed
                .get_mut(starting_polygon_index)
                .context("Bad polygon index")?
                .vertices
                .clone()
        } else {
            self.active
                .get_mut(final_polygon_index)
                .context("Bad polygon index")?
                .vertices
                .clone()
        };

        let mut iterator = vertices_clone.iter().enumerate().rev();
        while let Some((index, vertex)) = iterator.next() {
            match vertex.opening {
                Opening::Null
                | Opening::NewStart(_)
                | Opening::NewEnd(_)
                | Opening::Start(_)
                | Opening::End(_) => (),
                Opening::GenesisStart(_) => {
                    bail!("Dangling `Opening::GenesisStart`");
                }
                Opening::GenesisEnd(start) => {
                    let Some(next) = iterator.next() else {
                        bail!("`Opening::GenesisEnd` without a following vertex");
                    };
                    let Opening::GenesisStart(end) = next.1.opening else {
                        tracing::error!(
                            "Bad opening ({:?}) in polygon: {vertices_clone:#?}",
                            next.1.opening
                        );
                        bail!("`Opening::GenesisEnd` not followed by `Opening::GenesisStart`");
                    };

                    let next_index = next.0;

                    let joining_distances_range = end..start;
                    let joining_vertices_range = index..next_index;

                    let is_touching = self.join_final_polygon(
                        final_polygon_index,
                        joining_distances_range,
                        maybe_starting_polygon_index,
                        joining_vertices_range,
                    )?;

                    if is_touching && maybe_starting_polygon_index.is_some() {
                        self.handle_touching_polygons(final_polygon_index, None)?;
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Join a final polygon to either itself or a starting polygon.
    fn join_final_polygon(
        &mut self,
        base_polygon_index: usize,
        joining_opening_range: std::ops::Range<u32>,
        maybe_joining_polygon_index: Option<usize>,
        joining_vertices_range: std::ops::Range<usize>,
    ) -> Result<bool> {
        tracing::debug!(
            "Checking base polygon {base_polygon_index} \
             opening indices: {joining_vertices_range:?}"
        );

        let base_polygon = self
            .active
            .get_mut(base_polygon_index)
            .context("Bad polygon index")?;

        #[expect(
            clippy::indexing_slicing,
            reason = "A bad index would only be from bad code"
        )]
        let maybe_joining_polygon =
            maybe_joining_polygon_index.map(|index| &mut self.completed[index]);

        let Some(base_vertices_range) =
            super::super::vertices::find_contact(&base_polygon.vertices, &joining_opening_range)
        else {
            return Ok(false);
        };

        match maybe_joining_polygon {
            Some(joining_polygon) => {
                tracing::trace!("Final angle, polygon BEFORE: {:#?}", base_polygon);
                base_polygon.join_starting_polygon(
                    base_vertices_range,
                    joining_vertices_range,
                    joining_polygon,
                )?;
                tracing::trace!(
                    "Final angle, polygon AFTER joined polygon: {:#?}",
                    base_polygon
                );
            }
            None => {
                tracing::trace!("Final angle, self-polygon BEFORE: {:#?}", base_polygon);
                base_polygon.join_self(joining_vertices_range, base_vertices_range);
                tracing::trace!("Final angle, self-polygon AFTER: {:#?}", base_polygon);
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod test {
    use tvs_lib::ascii::assert_rasterised;

    use crate::joiner::{Joiner, rasterise_multi_polygon};

    #[test]
    fn final_segment_joins_starting_segment_simple() {
        crate::setup_logging();
        let part = vec![crate::segment::Segment::new(2, 2)];
        let joined =
            Joiner::join(&[part.clone(), vec![], vec![], vec![], vec![], part], 1.0).unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "████████████ ▀▀█████████",
            "████████████ ██▄▄▀▀█████",
            "████████████▄▀▀████ ████",
            "███████████████ ███ ████",
            "███████████████ ███ ████",
            "███████████████▄▀▀█ ████",
            "██████████████████▄▄████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn inherit_holes_final() {
        crate::setup_logging();
        let joined = Joiner::join(
            &[
                vec![crate::segment::Segment::new(0, 3)],
                vec![
                    crate::segment::Segment::new(0, 1),
                    crate::segment::Segment::new(2, 1),
                ],
                vec![crate::segment::Segment::new(0, 3)],
                vec![],
                vec![],
                vec![crate::segment::Segment::new(0, 5)],
            ],
            1.0,
        )
        .unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████ ▀▀█████████",
            "████████████ ██▄▄▀▀█████",
            "████████████ ██████▄ ▀██",
            "████████████ ████▀▄▄████",
            "████████████ ████ ██████",
            "██████████▀▀▄▀▀██ ██████",
            "███████▀▄▄██ ▀▀▄█ ██████",
            "████████▄▄▀▀▄▀▀▄▄███████",
            "████████████▄███████████",
            "████████████████████████",
            "████████████████████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn final_segment_joins_2_starting_segments() {
        crate::setup_logging();
        let long = vec![crate::segment::Segment::new(0, 4)];
        let starting = vec![
            crate::segment::Segment::new(0, 1),
            crate::segment::Segment::new(2, 1),
        ];
        let joined = Joiner::join(&[starting, vec![], vec![], vec![], vec![], long], 1.0).unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "████████████ ▀▀█████████",
            "████████████ ██▄▄▀▀█████",
            "████████████ ████▀▄▄████",
            "████████████ █▀ █ ██████",
            "████████████▄▀  █ ██████",
            "███████████████▄▀ ██████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn final_segment_joins_starting_segment_with_2_openings() {
        crate::setup_logging();
        let long = vec![crate::segment::Segment::new(0, 5)];
        let starting = vec![
            crate::segment::Segment::new(2, 1),
            crate::segment::Segment::new(4, 1),
        ];
        let joined =
            Joiner::join(&[starting, long.clone(), vec![], vec![], vec![], long], 1.0).unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████ ▀▀█████████",
            "████████████ ██▄▄▀▀█████",
            "████████████ ██████▄▄▀██",
            "████████████ ████▀▄ █ ██",
            "████████████ ▀▀ █ █ █ ██",
            "████████████ ▀▀ █ █ █ ██",
            "████████████ ██▄█ ▀ █ ██",
            "████████████ ██████▄█ ██",
            "████████████ ████▀▀▄▄███",
            "████████████ ▀▀▄▄███████",
            "████████████▄███████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn final_segment_joins_starting_segment_complex() {
        crate::setup_logging();
        let long = vec![crate::segment::Segment::new(0, 4)];
        let short = vec![crate::segment::Segment::new(2, 2)];
        let joined = Joiner::join(
            &[
                short.clone(),
                long.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                long,
                short,
            ],
            1.0,
        )
        .unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀▀▀▀▀▀▀████████",
            "█████████▄▀█████▄▀██████",
            "██████████▄▀██ ▀██▄▀████",
            "███████████▄▀ ██ ██ ████",
            "████████████ ▄▀▀ ██ ████",
            "█████████████ ██▄██ ████",
            "██████████████ ██▀▄█████",
            "███████████████ ▄███████",
            "████████████████████████",
            "████████████████████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn a_ring() {
        crate::setup_logging();
        let part = vec![crate::segment::Segment::new(2, 2)];
        let ring = vec![part; 10];
        let joined = Joiner::join(&ring, 1.0).unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀▀▄▄▄▀▀████████",
            "██████▀▄▄███████▄▄▀█████",
            "█████ ████▀▄▄▀▀███▄▀████",
            "████ ███▀▄█████▄▀███ ███",
            "████ ███ ███████ ███ ███",
            "████▄▀███▄▀▀██▀▄███▀▄███",
            "██████ █████▄▄████▀▄████",
            "███████▄▄▀▀███▀▀▄▄██████",
            "███████████▄▄▄██████████",
            "████████████████████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn rings() {
        crate::setup_logging();
        let part = vec![
            crate::segment::Segment::new(1, 1),
            crate::segment::Segment::new(3, 1),
        ];
        let ring = vec![part; 10];
        let joined = Joiner::join(&ring, 1.0).unwrap();
        let actual = rasterise_multi_polygon(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀▀▄▄▄▀▀████████",
            "██████▀▄ ▀▄▄▄▄▀▀ ▄▀█████",
            "█████ █ ██▀▀▄▄▀██ ▄▀████",
            "████ █ █▀▄▀▄▄▄▀▄▀█ █ ███",
            "████ █ █ █ ███ █ █ █ ███",
            "████▄▀▄▀█▄▀▄▄ ▀▄█▀▄▀▄███",
            "██████ ▄▀▀▀▄▄██▀▀▄▀▄████",
            "███████▄▄▀▀▄▄▄ ▀▄▄██████",
            "███████████▄▄▄██████████",
            "████████████████████████",
        ];
        assert_rasterised(&actual, &expected);
    }
}
