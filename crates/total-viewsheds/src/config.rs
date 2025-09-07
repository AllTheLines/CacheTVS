//! Defines all the CLI arguments.

use color_eyre::eyre::Result;

/// `Config`
#[derive(clap::Parser, Debug)]
#[clap(author, version)]
#[command(name = "tvs")]
#[command(
    about = "Generate _all_ the viewsheds for a given Digital Elevation Model, therefore the total viewsheds."
)]
pub struct Config {
    #[command(subcommand)]
    /// The subcommand.
    pub command: Commands,
}

/// CLI subcommand.
#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Run main computations.
    Compute(Compute),
    /// Reconstruct a viewshed.
    Viewshed(Viewshed),

    /// A hidden command that can be used to recursively print out all the subcommand help messages:
    ///   `cargo run dump-usage`
    /// Useful for updating the README.
    #[clap(hide(true))]
    DumpUsage,
}

/// Arguments to the `compute` subcommand.
#[derive(clap::Parser, Debug)]
pub struct Compute {
    /// The maximum distance in metres to search for visible points. For a TVS calculation to be
    /// truly correct, it must have access to all the DEM data around it that may possibly be
    /// visible to it. However, the further the distances searched the exponentially greater the
    /// computations required. Note that the largest currently known line of sight in the world
    /// is 538km. Defaults to one third of the DEM width.
    #[arg(long, value_name = "The maximum expected line of sight in meters")]
    pub max_line_of_sight: Option<u32>,

    // TODO: make this "reserved rings" and add support to the kernel so that the user can get
    // feedback of the actual number needed.
    //
    /// The maximum number of visible rings expected per km of band of sight. This is the number
    /// of times land may appear and disappear for an observer looking out into the distance. The
    /// value is used to decide how much memory is reserved for collecting ring data. So if it is
    /// too low then the program may panic. If it is too high then performance is lost due to
    /// unused RAM.
    #[arg(long, value_name = "Expected rings per km", default_value_t = 5.0)]
    pub rings_per_km: f32,

    /// The height of the observer in meters.
    #[arg(
        long,
        value_name = "Height of observer in meters",
        default_value = "1.65"
    )]
    pub observer_height: f32,

    /// Where to run the kernel calculations.
    #[arg(
        long,
        value_enum,
        value_name = "The method of running the kernel",
        default_value_t = Backend::Vulkan
    )]
    pub backend: Backend,

    /// Directory to save results in.
    #[arg(
        long,
        value_name = "Directory to save output to",
        default_value = "./output"
    )]
    pub output_dir: std::path::PathBuf,

    /// Override the calculated DEM points scale from the DEM file. Units in meters.
    #[arg(long, value_name = "DEM scale (meters)")]
    pub scale: Option<f64>,

    /// What to compute.
    #[arg(
        long,
        value_enum,
        value_name = "What to compute",
        value_delimiter = ',',
        default_value = "all"
    )]
    pub process: Vec<Process>,

    /// The input DEM file. Currently only `.hgt` files are supported.
    #[arg(value_name = "Path to the DEM file")]
    pub input: std::path::PathBuf,
}

#[derive(clap::Parser, Debug)]
pub struct Viewshed {
    /// Directory where compute output was saved.
    #[arg(value_name = "Path to existing output directory")]
    pub output_dir: std::path::PathBuf,

    /// Coordinates to reconstruct viewsheds for.
    #[arg(value_parser = parse_coords)]
    pub coordinates: Vec<(f32, f32)>,
}

fn parse_coords(string: &str) -> Result<(f32, f32)> {
    let mut coordinates = Vec::new();

    for coordinate in string.split(',') {
        coordinates.push(coordinate.parse::<f32>()?);
    }

    if coordinates.len() != 2 {
        color_eyre::eyre::bail!("Coordinate must be 2 numbers");
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "We already proved that the length is 2"
    )]
    Ok((coordinates[0], coordinates[1]))
}

/// Where to run the computations.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Backend {
    /// Conventional CPU computations. The slowest method.
    CPU,
    /// A SPIRV shader run on the GPU via Vulkan.
    Vulkan,
    /// TBC
    Cuda,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum Process {
    /// Calculate everything.
    All,
    /// Compute the total visible surfaves for each computable DEM point and output as a heatmap.
    TotalSurfaces,
    /// Find the longest line of sight for each computable point and output as a `GeoTiff`.
    LongestLineOfSight,
    /// Compute all the ring sectors saving them to disk so that they can be used to later
    /// reconstruct viewsheds.
    Viewsheds,
}
