//! We use the `.bt` filetype because it's simple and Rust doesn't yet have a good [`GeoTiff`] crate.
//! See: <http://vterrain.org/Implementation/Formats/BT.html>

/// The signature in the file header telling us this is a `.bt` file.
pub const MAGIC_BT_STRING: &str = "binterr1.3";

/// The of the header in bytes.
pub const HEADER_SIZE: u64 = 256;

/// DEM data is allowed to be in integers or floats.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum DataType {
    /// Integer data.
    #[default]
    Int16,
    /// Float data.
    Float32,
}

/// DEM data is allowed to be in integers or floats.
#[derive(Debug)]
pub enum Data {
    /// Integer data.
    Int16(std::vec::Vec<i16>),
    /// Float data.
    Float32(std::vec::Vec<f32>),
}

impl Default for Data {
    fn default() -> Self {
        Self::Int16(vec![0])
    }
}

/// The `.bt` file header.
#[expect(
    dead_code,
    reason = "I'd like to make this into its own crate at some point."
)]
#[derive(Default, Debug)]
pub struct Header {
    /// Width of DEM raster.
    pub width: u32,
    /// Height of DEM raster.
    pub height: u32,
    /// Bytes per elevation grid point, either 2 or 4.
    pub data_size: u16,
    /// Is the data floating point or not?
    pub data_type: DataType,
    /// Units of the distance between points:
    ///   0: Degrees
    ///   1: Meters
    ///   2: Feet (international foot = .3048 meters)
    ///   3: Feet (U.S. survey foot = 1200/3937 meters)
    pub horizontal_units: u16,
    /// Indicates the UTM zone (1-60) if the file is in UTM.
    /// Negative zone numbers are used for the southern hemisphere.
    pub utm_zone: u16,
    /// Indicates the Datum,
    ///
    /// The Datum field should be an EPSG Geodetic Datum Code, which are in the range of 6001 to 6904.
    /// If you are unfamiliar with these and do not care about Datum, you can simply use the value "6326"
    /// which is the WGS84 Datum.
    ///
    /// The simpler USGS Datum Codes are also supported for as a backward-compatibility with older files,
    /// but all new files should use the more complete EPSG Codes.
    pub datum: u16,
    /// The extents are specified in the coordinate space (projection) of the file. For example, if the
    /// file is using UTM, then the extents are in UTM coordinates.
    pub left: f64,
    /// The right-most extent.
    pub right: f64,
    /// The bottom-most extent.
    pub bottom: f64,
    /// The top-most extent.
    pub top: f64,
    /// Where to find the projection informaation.
    ///   0: Projection is fully described by this header
    ///   1: Projection is specified in a external .prj file
    pub projection_source: u16,
    /// Vertical units in meters, usually 1.0. The value 0.0 should be interpreted as 1.0 to allow
    /// for backward compatibility.
    pub vertical_scale: f32,
}

/// The `.bt` format layout.
pub struct BinaryTerrain {
    /// The header meta data.
    pub header: Header,
    /// The DEM data itself.
    pub data: Data,
}

#[expect(
    clippy::default_numeric_fallback,
    clippy::float_cmp,
    clippy::unreadable_literal,
    reason = "These are just tests"
)]
#[cfg(test)]
mod test {
    #[test]
    fn save_and_load() {
        let temporary_file = tempfile::NamedTempFile::new().unwrap().path().to_path_buf();
        let dem = crate::dem::DEM::new(
            crate::projection::LatLonCoord((-33.33, 45.67).into()),
            9,
            1.0,
            3,
        )
        .unwrap();
        #[rustfmt::skip]
        let total_surfaces = [
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0
        ];
        crate::output::bt::save(&dem, &total_surfaces, &temporary_file).unwrap();
        let bt = crate::bt::BinaryTerrain::read(&temporary_file).unwrap();
        assert_eq!(bt.header.width, 3);
        assert_eq!(bt.header.left, -33.330057749635486);
        assert_eq!(bt.header.right, -33.32994225028124);
        assert_eq!(bt.header.top, 45.669959512286916);
        assert_eq!(bt.header.bottom, 45.6700404876836);
        assert_eq!(bt.header.data_type, crate::bt::header::DataType::Float32);
        let crate::bt::header::Data::Float32(data) = bt.data else {
            panic!("`.bt` file's data type is not `f32`");
        };
        assert_eq!(data, total_surfaces);
    }
}
