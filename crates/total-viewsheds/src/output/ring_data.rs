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
    /// The small angular offset applied to each sector. See [`crate::dem::DEM`] for more details.
    pub sector_shift: f32,
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

#[cfg(test)]
mod test {
    use super::*;

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
}
