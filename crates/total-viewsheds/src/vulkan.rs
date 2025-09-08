//! GPU pipeline for running the kernel with Vulkan.
use wgpu::util::DeviceExt as _;

use color_eyre::Result;

/// Dimensions for the kernel. Used for workgroup sizes, dispatches and invocations.
type Dimensions = (u32, u32, u32);

/// Size of each kernel workgroup. MUST MATCH `compute(threads(8, 8, 4)` in the kernel.
const KERNEL_WORKGROUPS: Dimensions = (8, 8, 4);

/// Manage the Vulkan-enabled GPU.
pub struct Vulkan {
    /// The GPU device.
    device: wgpu::Device,
    /// The pipeline's command queue.
    queue: wgpu::Queue,
    /// Number of times to call each kernel workgroup.
    dispatches: Dimensions,

    /// Buffers of data needed for computations.
    buffers: Buffers,
    /// How the buffers get passed as arguments to the kernel.
    bind_group: wgpu::BindGroup,

    /// The GPU pipeline.
    pipeline: wgpu::ComputePipeline,

    /// Memory size of all the accumulated surface data.
    output_surfaces_size: u64,
    /// Memory size of the ring data (raw viewshed data).
    output_rings_size: u64,
}

/// All the buffers used in the pipeline.
struct Buffers {
    /// The horizontal distance of each DEM point from each other in the sector. So it changes for
    /// every sector (therefore angle).
    distances: wgpu::Buffer,
    /// Deltas that when applied to the indexe IDs in a DEM allow traversing from the point of view
    /// along the band of sight.
    band_deltas: wgpu::Buffer,
    /// GPU-side buffer to save the accumulated visible surface area for a given point.
    output_surfaces: wgpu::Buffer,
    /// CPU-side buffer for the above.
    download_surfaces: wgpu::Buffer,
    /// GPU-side buffer to store the actual viewshed extent data.
    output_rings: wgpu::Buffer,
    /// CPU-side buffer for the above.
    download_rings: wgpu::Buffer,
}

impl Vulkan {
    /// Instantiate.
    pub fn new(
        mut constants: kernel::constants::Constants,
        elevations: Vec<f32>,
        distances_count: usize,
        band_deltas_count: usize,
        total_reserved_rings: usize,
    ) -> Result<Self> {
        // We first initialize an wgpu `Instance`, which contains any "global" state wgpu needs.
        //
        // This is what loads the vulkan/dx12/metal/opengl libraries.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
        });

        // We then create an `Adapter` which represents a physical gpu in the system. It allows
        // us to query information about it and create a `Device` from it.
        //
        // This function is asynchronous in WebGPU, so request_adapter returns a future. On native/webgl
        // the future resolves immediately, so we can block on it without harm.
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

        let limits = adapter.limits();
        tracing::debug!("GPU limits: {limits:?}");

        // Print out some basic information about the adapter.
        tracing::info!("Running on GPU adapter: {:#?}", adapter.get_info());

        // Check to see if the adapter supports compute shaders. While WebGPU guarantees support for
        // compute shaders, wgpu supports a wider range of devices through the use of "downlevel" devices.
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        if !downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            color_eyre::eyre::bail!("Adapter does not support compute shaders");
        }

        // We then create a `Device` and a `Queue` from the `Adapter`.
        //
        // The `Device` is used to create and manage GPU resources.
        // The `Queue` is a queue used to submit work for the GPU to process.
        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 5,
            max_storage_buffer_binding_size: 2_000_000_000,
            max_buffer_size: 2_000_000_000,
            max_compute_workgroups_per_dimension: 1024,
            ..wgpu::Limits::default()
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                required_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            }))?;

        let distances_size =
            u64::try_from(distances_count)? * u64::try_from(std::mem::size_of::<f32>())?;
        let band_deltas_size =
            u64::try_from(band_deltas_count)? * u64::try_from(std::mem::size_of::<i32>())?;
        let output_surfaces_size = u64::from(constants.total_bands.div_euclid(2))
            * u64::try_from(std::mem::size_of::<f32>())?;
        let output_rings_size =
            u64::try_from(total_reserved_rings)? * u64::try_from(std::mem::size_of::<u32>())?;

        let required_invocations = constants.total_bands * 2;
        let (dispatches, invocations) =
            Self::find_dispatch_dimensions(required_invocations, KERNEL_WORKGROUPS)?;
        constants.dimensions = [
            invocations.0,
            invocations.1,
            invocations.2,
            required_invocations,
        ]
        .into();

        let (buffers, bind_group) = Self::setup_buffers(
            &device,
            constants,
            elevations,
            distances_size,
            band_deltas_size,
            output_rings_size,
        )?;

        // TODO: embed this if we ever make a proper binary release.
        // Safety:
        //   We're disabling Naga vailidation, so we must assume that if `cargo-gpu` can compile
        //   the Rust shader then we have a valid SPIRV file.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::include_spirv_raw!(
                "../../kernels/vulkan-and-cpu//kernel.spv"
            ))
        };

        // The pipeline layout describes the bind groups that a pipeline expects
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&Self::create_bind_group_layout(&device)?],
            push_constant_ranges: &[],
        });

        // The pipeline is the ready-to-go program state for the GPU. It contains the shader modules,
        // the interfaces (bind group layouts) and the shader entry point.
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let gpu = Self {
            device,
            queue,
            dispatches,
            buffers,
            bind_group,
            pipeline,
            output_surfaces_size,
            output_rings_size,
        };

        tracing::trace!("GPU pipline ready.");
        Ok(gpu)
    }

    /// Setup the buffers.
    fn setup_buffers(
        device: &wgpu::Device,
        constants: kernel::constants::Constants,
        elevations: Vec<f32>,
        distances_size: u64,
        band_deltas_size: u64,
        output_rings_size: u64,
    ) -> Result<(Buffers, wgpu::BindGroup)> {
        tracing::trace!("Creating GPU buffers....");
        let input_constants_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Constants"),
            contents: bytemuck::bytes_of(&constants),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // All the elevation data.
        let input_elevations_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Elevations"),
                contents: bytemuck::cast_slice(&elevations),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // A buffer to upload the distance data for each sector.
        let input_distances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Distances"),
            size: distances_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // A buffer to upload the band deltas for each sector.
        let input_deltas_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Band deltas"),
            size: band_deltas_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_surfaces_size = u64::from(constants.total_bands.div_euclid(2))
            * u64::try_from(std::mem::size_of::<f32>())?;
        let output_surfaces_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output accumulated surfaces"),
            size: output_surfaces_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Finally we create buffers which can be read by the CPU. This buffer is how we will read
        // the data. We need to use a separate buffer because we need to have a usage of `MAP_READ`,
        // and that usage can only be used with `COPY_DST`.
        let download_surfaces_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download surfaces"),
            size: output_surfaces_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let output_rings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output ring data"),
            size: output_rings_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download_rings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download ring data"),
            size: output_rings_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = Self::create_bind_group_layout(device)?;
        // The bind group contains the actual resources to bind to the pipeline.
        //
        // Even when the buffers are individually dropped, wgpu will keep the bind group and buffers
        // alive until the bind group itself is dropped.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_constants_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_elevations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_distances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_deltas_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_surfaces_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_rings_buffer.as_entire_binding(),
                },
            ],
        });

        let buffers = Buffers {
            distances: input_distances_buffer,
            band_deltas: input_deltas_buffer,
            output_surfaces: output_surfaces_buffer,
            download_surfaces: download_surfaces_buffer,
            output_rings: output_rings_buffer,
            download_rings: download_rings_buffer,
        };

        drop(elevations); // Free up RAM. Although it gets dropped anyway right??

        tracing::trace!("...GPU buffers created.");
        Ok((buffers, bind_group))
    }

    /// A bind group layout describes the types of resources that a bind group can contain. Think
    /// of this like a C-style header declaration, ensuring both the pipeline and bind group agree
    /// on the types of resources.
    fn create_bind_group_layout(device: &wgpu::Device) -> Result<wgpu::BindGroupLayout> {
        let constants_size = u64::try_from(std::mem::size_of::<kernel::constants::Constants>())?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Constants
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(constants_size),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Elevations
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Distances
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Deltas
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output: surface data
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output: ring data
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        Ok(layout)
    }

    /// Compute a single sector.
    pub fn run(&self, distances: &[f32], band_deltas: &[i32]) -> Result<(Vec<f32>, Vec<u32>)> {
        self.queue
            .write_buffer(&self.buffers.distances, 0, bytemuck::cast_slice(distances));
        self.queue.write_buffer(
            &self.buffers.band_deltas,
            0,
            bytemuck::cast_slice(band_deltas),
        );

        // The command encoder allows us to record commands that we will later submit to the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // A compute pass is a single series of compute operations. While we are recording a compute
        // pass, we cannot record to the encoder.
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(self.dispatches.0, self.dispatches.1, self.dispatches.2);

        // Now we drop the compute pass, giving us access to the encoder again.
        drop(compute_pass);

        // We add a copy operation to the encoder. This will copy the data from the output buffer on the
        // GPU to the download buffer on the CPU.
        encoder.copy_buffer_to_buffer(
            &self.buffers.output_surfaces,
            0,
            &self.buffers.download_surfaces,
            0,
            self.output_surfaces_size,
        );

        encoder.copy_buffer_to_buffer(
            &self.buffers.output_rings,
            0,
            &self.buffers.download_rings,
            0,
            self.output_rings_size,
        );

        // We finish the encoder, giving us a fully recorded command buffer.
        let command_buffer = encoder.finish();

        // At this point nothing has actually been executed on the gpu. We have recorded a series of
        // commands that we want to execute, but they haven't been sent to the gpu yet.
        //
        // Submitting to the queue sends the command buffer to the gpu. The gpu will then execute the
        // commands in the command buffer in order.
        self.queue.submit([command_buffer]);

        // We now map the download buffers so we can read it. Mapping tells wgpu that we want to read/write
        // to the buffer directly by the CPU and it should not permit any more GPU operations on the buffer.
        //
        // Mapping requires that the GPU be finished using the buffer before it resolves, so mapping has a callback
        // to tell you when the mapping is complete.
        let buffer_slice_surfaces = self.buffers.download_surfaces.slice(..);
        buffer_slice_surfaces.map_async(wgpu::MapMode::Read, |_| {
            // In this case we know exactly when the mapping will be finished,
            // so we don't need to do anything in the callback.
        });
        let buffer_slice_rings = self.buffers.download_rings.slice(..);
        buffer_slice_rings.map_async(wgpu::MapMode::Read, |_| {});

        // Wait for the GPU to finish working on the submitted work. This doesn't work on WebGPU, so we would need
        // to rely on the callback to know when the buffer is mapped.
        self.device.poll(wgpu::PollType::Wait)?;

        // We can now read the data from the buffer.
        let surfaces_data = buffer_slice_surfaces.get_mapped_range();
        let ring_data = buffer_slice_rings.get_mapped_range();
        // Convert the data back to a slice of f32.
        let surfaces_result = bytemuck::cast_slice(&surfaces_data).to_vec();
        let ring_result = bytemuck::cast_slice(&ring_data).to_vec();

        drop(surfaces_data);
        drop(ring_data);

        self.buffers.download_surfaces.unmap();
        self.buffers.download_rings.unmap();

        Ok((surfaces_result, ring_result))
    }

    /// Find a 3D kernel dispatch that balances all dimensions.
    pub fn find_dispatch_dimensions(
        total_invocations: u32,
        workgroups: Dimensions,
    ) -> Result<(Dimensions, Dimensions)> {
        let max_tries = 1_000_000u32;
        let mut dispatches = [0u32, 0, 0];
        let mut invocations: Dimensions;
        let mut total_invocations_generated;
        for _ in 0..max_tries {
            for dimension in 0..3 {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "The loop range doesn't fall outside the arra size"
                )]
                {
                    dispatches[dimension] += 1;
                };

                invocations = (
                    dispatches[0] * workgroups.0,
                    dispatches[1] * workgroups.1,
                    dispatches[2] * workgroups.2,
                );

                total_invocations_generated = invocations.0 * invocations.1 * invocations.2;
                if total_invocations_generated >= total_invocations {
                    tracing::debug!(
                    "Kernel dimensions (for {workgroups:?}). \
                    Dispatches: {dispatches:?}. Invocations: {invocations:?} (total: {total_invocations_generated}, needed: {total_invocations} )"
                );

                    return Ok((dispatches.into(), invocations));
                }
            }
        }

        color_eyre::eyre::bail!("Couldn't find GPU dispatch dimensions");
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn find_dispatches() {
        let (dispatches, invocations) =
            Vulkan::find_dispatch_dimensions(1_000_000, KERNEL_WORKGROUPS).unwrap();
        assert_eq!(dispatches, (16, 16, 16));
        assert_eq!(invocations, (128, 128, 64));
    }
}
