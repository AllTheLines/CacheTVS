//! The raw, underlying data used to reconstruct viewsheds.

use color_eyre::{eyre::ContextCompat as _, Result};

/// Name of the `fjall` partition.
const PARTITION_NAME: &str = "ring_data";

/// The key name for the metadata.
const METADATA_KEY: &str = "metadata";

/// Whether the data is coming from disk or RAM.
pub enum Source {
    /// The path to the data on disk.
    Directory(std::path::PathBuf),
    #[cfg(test)]
    RAM(AllData),
}

/// Whether the data represents all possible angles (sectors), or just a single angle.
pub enum SectorData {
    /// Data represents all sectors.
    #[cfg(test)]
    AllSectors(Vec<Vec<u32>>),
    /// Data only represents a single sector.
    Sector(Storage),
}

/// All the data. Includes both sector data and metadata.
pub struct AllData {
    /// Metadata for the data.
    pub metadata: MetaData,
    /// The actual data by organised by sectors.
    pub ring_data: SectorData,
}

impl AllData {
    /// Instantiate with data from disk.
    pub fn new_from_storage(output_directory: &std::path::Path) -> Result<Self> {
        let storage = Storage::new(output_directory)?;
        let metadata = storage.load_metadata()?;
        Ok(Self {
            metadata,
            ring_data: SectorData::Sector(storage),
        })
    }

    /// Get a single sector of data.
    pub fn get_sector(&self, angle: u16) -> Result<Vec<u32>> {
        match &self.ring_data {
            #[cfg(test)]
            SectorData::AllSectors(items) => Ok(items
                .get(usize::from(angle))
                .context("Couldn't find sector data.")?
                .clone()),
            SectorData::Sector(storage) => {
                let sector = storage.load_sector(angle)?;
                Ok(sector)
            }
        }
    }
}

/// Metadata about the main data.
#[derive(serde::Serialize, serde::Deserialize, Default, Debug, Clone)]
pub struct MetaData {
    /// The width of the 2D grid of elevation data. The algorithm requires that the grid be square,
    /// so there is no need for a height field.
    pub width: u32,
    /// The diameter in meters each point of the data covers.
    pub scale: f32,
    /// The maximum line of sight that was used to calculate the ring data. It is needed to
    /// instantiate the `DEM` struct and therefore reconstruct the bands of sight used to create
    /// the ring data.
    pub max_line_of_sight: u32,
    /// The number of items reserved to place ring DEM IDs in.
    pub reserved_ring_size: usize,
    /// The lat/lon coordinates for the centre of the 2D DEM grid. Used for accurately converting
    /// between degree and metric coordinate systems.
    pub centre: crate::projection::LatLonCoord,
}

pub struct Storage {
    /// An active handle to the database.
    db: fjall::PartitionHandle,

    /// See: <https://github.com/fjall-rs/fjall/issues/183>
    _keyspace: fjall::Keyspace,
}

impl Storage {
    /// Instantitate.
    pub fn new(output_directory: &std::path::Path) -> Result<Self> {
        let ring_data_directory = output_directory.join("ring_data");
        let keyspace = fjall::Config::new(ring_data_directory).open()?;
        let db =
            keyspace.open_partition(PARTITION_NAME, fjall::PartitionCreateOptions::default())?;

        Ok(Self {
            _keyspace: keyspace,
            db,
        })
    }

    /// The key to database record for the given sector.
    fn angle_key(angle: u16) -> String {
        format!("{angle}")
    }

    /// Load the metadata.
    pub fn load_metadata(&self) -> Result<MetaData> {
        tracing::debug!("Loading metadata from {:?}...", self.db.path());

        let metadata_bytes = self
            .db
            .get(METADATA_KEY)?
            .context("Couldn't find ring data metadata.")?;
        let metadata = serde_json::from_slice(&metadata_bytes)?;
        tracing::info!("Loaded metadata: {metadata:?}");

        Ok(metadata)
    }

    /// Save the metadata.
    pub fn save_metadata(&self, metadata: &MetaData) -> Result<()> {
        tracing::debug!("Saving metadata...");
        let start = std::time::Instant::now();

        let serialised = serde_json::to_string(metadata)?;
        self.db.insert(METADATA_KEY, &serialised)?;

        tracing::debug!("...saved in {:?}ms", start.elapsed().as_millis());
        Ok(())
    }

    /// Load ring data for a single sector.
    pub fn load_sector(&self, angle: u16) -> Result<Vec<u32>> {
        tracing::debug!("Loading ring data from {:?}...", self.db.path());

        let sector_bytes = self
            .db
            .get(Self::angle_key(angle))?
            .context(format!("Couldn't find sector {angle} in storage."))?;
        let sector: Vec<u32> = bytemuck::cast_slice(&sector_bytes).to_vec();

        Ok(sector)
    }

    /// Save sector data for a single sector.
    pub fn save_sector(&self, angle: u16, ring_data: &[u32]) -> Result<()> {
        tracing::debug!(
            "Saving ring data ({} items) for sector {angle}...",
            ring_data.len()
        );
        let start = std::time::Instant::now();

        let data: &[u8] = bytemuck::cast_slice(ring_data);
        self.db.insert(Self::angle_key(angle), data)?;

        tracing::debug!("...saved in {:?}ms", start.elapsed().as_millis());
        Ok(())
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "Not sure of the best approach right now."
)]
/// Convert a bitmap of visibilities to an array of opening/closing IDs. Or in other words
/// converting the ring data of the CPU kernel to the same format of the ring data of the GPU
/// kernel. There is no inherent reason that the formats should be different, so we should decide
/// on one over the other, likely the bitmap.
///
/// The CPU algorithm handles one angle at a time, so this will be called 360 times. But the GPU
/// algorighm handles one _sector_ at a time, so is only called 180 times. It combines opposite
/// angles into forward and backward lines of sight.
///
/// # Input CPU format
///   * Array shape: `[ [boolean visibility; max_line_of_sight]; tvs_points_count ]`.
///
/// [
///   // Point 0,0:
///   [
///     true, true, false, false, true, false,
///   ],
///
///   // Point 1,0:
///   [
///     true, true, true, true, true, false,
///   ],
///
///   ...
/// ]
///
///
/// # Output GPU format
///   * Array is of length `tvs_points_count * reserved_ring_data_size`.
///   * Ring data size in this example is 10.
///
/// Note how the first ID is always a closing ID because we can always assume that the point from
/// where the observer stands is always visible.
///
///   | Size of used reserved ring data | closing | opening | closing ...
/// [
///
///     // Point 0,0:
///     4,                                2,        4,        5,        0,0,0,0,0,0,
///     // Point 1,0:
///     2,                                5,                            0,0,0,0,0,0,0,0
///
///   ...
///
///   // Then the whole thing is repeated for "backward" lines of sight.
/// ]
pub fn convert_bitmap_to_ids(
    bitmap: &[Vec<bool>],
    reserved_ring_data_size: usize,
    sector: usize,
    width: u32,
) -> Result<Vec<u32>> {
    let total_points = usize::try_from(width * width)?;
    let mut ring_data = vec![0; total_points * reserved_ring_data_size];
    let mut start = 0;

    for (rotated_tvs_id, line_of_sight) in bitmap.iter().enumerate() {
        let mut is_currently_visible = true;
        let mut is_previously_visible = true;
        let mut closing = false;
        let mut cursor = 1;

        if width != 0 {
            let tvs_id = kernel::rotation::Rotator::new_from_angle(
                rotated_tvs_id as u32,
                width,
                sector as f32,
            )
            .rotate_dem_id();

            if tvs_id == 5 {
                // dbg!(rotated_tvs_id, line_of_sight);
            }
        }
        for (index, visibility) in line_of_sight
            .iter()
            // Skip the first visibility because we assume the PoV is always visibile
            .skip(1)
            .enumerate()
        {
            is_currently_visible = *visibility;
            let opening = is_currently_visible && !is_previously_visible;
            closing = is_previously_visible && !is_currently_visible;

            if opening || closing {
                ring_data[start + cursor] = u32::try_from(index + 1)?;
                cursor += 1;
            }

            is_previously_visible = is_currently_visible;
        }

        if is_currently_visible && !closing {
            ring_data[start + cursor] = u32::try_from(width)?;
            cursor += 1;
        }

        ring_data[start] = u32::try_from(cursor)?;
        start += reserved_ring_data_size;
    }

    Ok(ring_data)
}

#[cfg(test)]
mod test {
    use super::*;
    use googletest::prelude::*;

    #[test]
    fn save_and_load() {
        let temporary_directory = tempfile::tempdir().unwrap();
        let directory = temporary_directory.path();
        let storage = Storage::new(directory).unwrap();
        storage.save_sector(0, &[42]).unwrap();
        let metadata = MetaData {
            width: 69,
            ..MetaData::default()
        };
        storage.save_metadata(&metadata).unwrap();
        let all_data = AllData::new_from_storage(directory).unwrap();
        assert_eq!(all_data.metadata.width, 69);
        match all_data.ring_data {
            SectorData::AllSectors(_) => panic!("Expected `SectorData::Sector(_)`"),
            SectorData::Sector(ring_data) => {
                assert_eq!(ring_data.load_sector(0).unwrap(), vec![42]);
            }
        }
    }

    #[gtest]
    fn convert_bitmap() {
        fn run(bitmap: &[Vec<bool>]) -> Vec<u32> {
            let reserved_ring_size = 3;
            let result = convert_bitmap_to_ids(bitmap, reserved_ring_size, 0, 4).unwrap();
            let size = bitmap.len() * reserved_ring_size;
            result[0..size].to_vec()
        }

        expect_eq!(run(&[vec![true, true, true, false]]), vec![2, 3, 0]);

        expect_eq!(run(&[vec![true, false, false, false]]), vec![2, 1, 0]);

        expect_eq!(
            run(&[
                vec![true, true, true, false],
                vec![true, false, false, false]
            ]),
            vec![2, 3, 0, 2, 1, 0]
        );

        expect_eq!(run(&[vec![true, true, true, true]]), vec![2, 4, 0]);
    }
}
