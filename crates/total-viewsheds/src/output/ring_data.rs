//! The raw, underlying data used to reconstruct viewsheds.

use color_eyre::{Result, eyre::ContextCompat as _};

/// Name of the `fjall` partition.
const PARTITION_NAME: &str = "ring_data";

/// The key name for the metadata.
const METADATA_KEY: &str = "metadata";

/// Whether the data is coming from disk or RAM.
pub enum Source {
    /// The path to the data on disk in flat files.
    Directory(std::path::PathBuf),
    /// The path to the data on disk in an Sqlite DB.
    SQLite(std::path::PathBuf),
    #[cfg(test)]
    RAM(AllData),
}

/// Whether the data represents all possible angles (sectors), or just a single angle.
pub enum SectorData {
    /// Data represents all sectors.
    #[cfg(test)]
    AllSectors(Vec<Vec<u32>>),
    /// Data only represents a single sector from Fjall.
    FjallSector(FjallStorage),
    /// Data represents all segemnts for a given DEM ID.
    SQLiteSegments,
}

/// All the data. Includes both sector data and metadata.
pub struct AllData {
    /// Metadata for the data.
    pub metadata: crate::cpu::storage::metadata::MetaData,
    /// The actual data by organised by sectors.
    pub ring_data: SectorData,
}

impl AllData {
    /// Instantiate with data from flat files.
    pub fn new_from_fjall(output_directory: &std::path::Path) -> Result<Self> {
        let storage = FjallStorage::new(output_directory)?;
        let metadata = storage.load_metadata()?;
        Ok(Self {
            metadata,
            ring_data: SectorData::FjallSector(storage),
        })
    }

    /// Instantiate with data from Sqlite.
    pub fn new_from_sqlite(db_path: &std::path::Path) -> Result<Self> {
        let db = crate::cpu::storage::db::DB::new(db_path)?;
        let metadata = db.load_metadata()?;
        Ok(Self {
            metadata,
            ring_data: SectorData::SQLiteSegments,
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
            SectorData::FjallSector(storage) => {
                let sector = storage.load_sector(angle)?;
                Ok(sector)
            }
            SectorData::SQLiteSegments => {
                color_eyre::eyre::bail!(
                    "Our Sqlite implementation can be queried directly by DEM ID"
                )
            }
        }
    }
}

/// In-memory DB for Vulkan's ring data storage. To be deprecated soon
pub struct FjallStorage {
    /// An active handle to the database.
    db: fjall::PartitionHandle,

    /// See: <https://github.com/fjall-rs/fjall/issues/183>
    _keyspace: fjall::Keyspace,
}

impl FjallStorage {
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
    pub fn load_metadata(&self) -> Result<crate::cpu::storage::metadata::MetaData> {
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
    pub fn save_metadata(&self, metadata: &crate::cpu::storage::metadata::MetaData) -> Result<()> {
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
    fn save_and_load_fjall() {
        let temporary_directory = tempfile::tempdir().unwrap();
        let directory = temporary_directory.path();
        let storage = FjallStorage::new(directory).unwrap();
        storage.save_sector(0, &[42]).unwrap();
        let metadata = crate::cpu::storage::metadata::MetaData {
            width: 69,
            ..crate::cpu::storage::metadata::MetaData::default()
        };
        storage.save_metadata(&metadata).unwrap();
        let all_data = AllData::new_from_fjall(directory).unwrap();
        assert_eq!(all_data.metadata.width, 69);
        match all_data.ring_data {
            SectorData::AllSectors(_) | SectorData::SQLiteSegments => {
                panic!("We're only testing Fjall")
            }
            SectorData::FjallSector(ring_data) => {
                assert_eq!(ring_data.load_sector(0).unwrap(), vec![42]);
            }
        }
    }
}
