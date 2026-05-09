//! Defines all the CLI arguments.

use color_eyre::eyre::{Result, bail};
use std::path::PathBuf;

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
    /// Post-processing tasks
    PostProcess(PostProcess),

    /// A hidden command that can be used to recursively print out all the subcommand help messages:
    ///   `cargo run dump-usage`
    /// Useful for updating the README.
    #[clap(hide(true))]
    DumpUsage,
}

/// Arguments to the `compute` subcommand.
#[derive(clap::Parser, Debug, Default)]
pub struct Compute {
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
        default_value_t = Backend::CPU
    )]
    pub backend: Backend,

    /// Directory to save results in.
    #[arg(
        long,
        value_name = "Directory to save output to",
        default_value = "./output"
    )]
    pub output_dir: std::path::PathBuf,

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

    /// How to normalise heatmap data
    #[arg(
        long,
        value_enum,
        value_name = "Heatmap normalisation method",
        default_value_t = HeatmapNormalisation::Exponential
    )]
    pub heatmap: HeatmapNormalisation,

    /// Air refraction coefficient. Therefore, how much impact refraction has on visibility. Values
    /// typically range from 0.1 (less impact) to 0.2 (more impact).
    #[arg(
        long,
        value_name = "Air refraction coefficient",
        default_value = "0.13"
    )]
    pub refraction: f32,

    /// Thread count used for CPU parallelism
    #[arg(long, value_name = "Thread count", default_value = "8")]
    pub thread_count: usize,

    /// Controls line of sight and total viewshed image generation
    #[arg(long, value_name = "Render image", default_value = "false")]
    pub disable_image_render: bool,

    /// Where to store the viewshed data. Requires build with `--features=ring_data`.
    #[arg(
        long,
        value_name = "viewshed storage path",
        default_value = "./viewsheds.db"
    )]
    pub viewsheds_db_path: PathBuf,

    /// Lon/lat coordinates for a polygon that represents an Area of Interest within the DEM.
    /// Points outside this polygon will be ignored.
    #[arg(
        long,
        allow_hyphen_values(true),
        value_parser = parse_coords,
        value_name = "Lon/lat coords of region")
    ]
    pub aoi_point: Vec<(f32, f32)>,

    /// Creates a database per thread, managed by an additional worker thread
    #[arg(
        long,
        value_name = "aggregate data into a database per-thread",
        default_value = "false"
    )]
    pub database_per_thread: bool,

    /// Provide an existing TVS heatmap .tiff as a source for directing computation.
    #[arg(long, value_name = "Path to existing TVS .tiff")]
    pub tvs_source_path: Option<std::path::PathBuf>,

    /// Only save viewsheds with the highest surface area within a square of this size. Must be a
    /// perfect square, eg: 4, 9, 16, etc. Requires `--tvs_source_path` argument.
    #[arg(long, value_name = "Square size", value_parser=is_perfect_square)]
    pub only_save_biggest_viewsheds: Option<u16>,
}

/// Arguments to the `viewshed` subcommand.
#[derive(clap::Parser, Debug)]
pub struct Viewshed {
    /// Path or directory where viewshed DB was saved.
    #[arg(value_name = "Path to existing database")]
    pub db_path: std::path::PathBuf,

    /// Directory to save viewshed in.
    #[arg(value_name = "Viewshed output directory")]
    pub output_dir: std::path::PathBuf,

    /// Coordinates to reconstruct viewsheds for.
    #[arg(value_parser = parse_coords)]
    pub coordinates: Vec<(f32, f32)>,
}

/// Arguments to the `post-process` subcommand.
#[derive(clap::Parser, Debug)]
pub struct PostProcess {
    /// Directory where viewshed DBs were saved.
    #[arg(value_name = "Path to existing databases")]
    pub db_dir: std::path::PathBuf,

    /// Number of threads to use. Defaults to letting Rayon choose the thread count.
    #[arg(long, value_name = "Number of threads to use")]
    pub threads: Option<usize>,
}

/// Parse args like "123.456,-987.654" to floats.
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

/// Check if argument is a perfect square.
fn is_perfect_square(argument: &str) -> Result<u16> {
    let number = argument.parse::<u16>()?;
    let root = number.isqrt();
    if root * root != number {
        bail!("Value must be a perfect square");
    }
    Ok(number)
}

/// Where to run the computations.
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum Backend {
    /// Optimised cache-efficient CPU kernel
    #[default]
    CPU,
    /// TBC
    Cuda,
}

/// Which calculations to process.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum Process {
    /// Calculate everything.
    All,
    /// Compute all the polar segments saving them to disk so that they can be used to later
    /// reconstruct viewsheds.
    Viewsheds,
}

/// Where to run the computations.
#[derive(clap::ValueEnum, Clone, Debug, Copy, Default)]
pub enum HeatmapNormalisation {
    /// Just scale between 0 and 1
    UnitScale,
    /// Scale between 0 and 1 with an exponential factor.
    #[default]
    Exponential,
    #[expect(clippy::doc_markdown, reason = "This is displayed on the CLI")]
    /// Use Z-score normalisation based on Welford's algorithm. This basically means that the
    /// data is redistributed such that the mean is 0.5. Useful for overly dark or bright
    /// heatmaps.
    /// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    Welford,
}
