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

    /// Constants for the computations.
    constants: kernel::constants::Constants,

    /// Rotation compute resources
    rotation: ComputePassResources,
    /// Visibility compute resources
    visibility: ComputePassResources,

    /// Buffers of data for returning data to the CPU.
    output_buffers: OutputBuffers,
    /// Memory size of all the accumulated surface data.
    output_surfaces_size: u64,
    /// Memory size of the ring data (raw viewshed data).
    output_rings_size: u64,
    /// Memory size of the longest lines of sight data.
    output_longest_lines_size: u64,
}

/// All the buffers used in the pipeline.
struct OutputBuffers {
    /// GPU-side buffer to save the accumulated visible surface area for a given point.
    output_surfaces: wgpu::Buffer,
    /// CPU-side buffer for the above.
    download_surfaces: wgpu::Buffer,
    /// GPU-side buffer to store the actual viewshed extent data.
    output_rings: wgpu::Buffer,
    /// CPU-side buffer for the above.
    download_rings: wgpu::Buffer,
    /// GPU-side buffer to save the longest line of sight for a given point.
    output_longest_lines: wgpu::Buffer,
    /// CPU-side buffer for the above.
    download_longest_lines: wgpu::Buffer,
}

/// Resources used in the GPU pipelines.
struct ComputePassResources {
    /// Uniform buffers for constants and the like.
    uniform: wgpu::Buffer,
    /// The dimensions of kernel invocations.
    dispatches: Dimensions,
    /// The constants for sending to the uniforms.
    constants: kernel::constants::Constants,
    /// The GPU pipeline definitions.
    pipeline: wgpu::ComputePipeline,
    /// How arguments are defined for the shader entry points.
    bindgroup: wgpu::BindGroup,
}

impl Vulkan {
    /// Instantiate.
    pub fn new(
        constants: kernel::constants::Constants,
        elevations: Vec<i16>,
        chocolate_box_size: usize,
        total_reserved_rings: usize,
    ) -> Result<Self> {
        let instance = Self::instance();

        // We then create an `Adapter` which represents a physical gpu in the system. It allows
        // us to query information about it and create a `Device` from it.
        //
        // This function is asynchronous in WebGPU, so request_adapter returns a future. On native/webgl
        // the future resolves immediately, so we can block on it without harm.
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;

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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
                required_limits: Self::limits(&adapter),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                // Safety:
                //   We're using `EXPERIMENTAL_PASSTHROUGH_SHADERS`.
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            }))?;

        let total_bands = u64::from(constants.total_bands.div_euclid(2));
        let rotations_size =
            u64::try_from(chocolate_box_size)? * u64::try_from(std::mem::size_of::<f32>())?;
        let output_surfaces_size = total_bands * u64::try_from(std::mem::size_of::<f32>())?;
        let output_rings_size =
            u64::try_from(total_reserved_rings)? * u64::try_from(std::mem::size_of::<u32>())?;
        let output_longest_lines_size = total_bands * u64::try_from(std::mem::size_of::<f32>())?;

        let required_rotation_invocations = u32::try_from(chocolate_box_size)?;
        let (rotation_dispatches, rotation_invocations) =
            Self::find_dispatch_dimensions(required_rotation_invocations, KERNEL_WORKGROUPS)?;
        let mut rotation_constants = constants;
        rotation_constants.dimensions = [
            rotation_invocations.0,
            rotation_invocations.1,
            rotation_invocations.2,
            required_rotation_invocations,
        ]
        .into();
        let (rotated_elevations_buffer, rotations_constants_buffer, rotations_bind_group) =
            Self::setup_rotation_buffers(&device, elevations, rotations_size)?;

        let required_visibility_invocations = constants.total_bands * 2;
        let (visibility_dispatches, visibility_invocations) =
            Self::find_dispatch_dimensions(required_visibility_invocations, KERNEL_WORKGROUPS)?;
        let mut visibility_constants = constants;
        visibility_constants.dimensions = [
            visibility_invocations.0,
            visibility_invocations.1,
            visibility_invocations.2,
            required_visibility_invocations,
        ]
        .into();
        let (output_buffers, visibility_constants_buffer, visibility_bind_group) =
            Self::setup_visibility_buffers(
                &device,
                constants,
                &rotated_elevations_buffer,
                output_rings_size,
                output_longest_lines_size,
            )?;

        let rotation = ComputePassResources {
            uniform: rotations_constants_buffer,
            dispatches: rotation_dispatches,
            constants: rotation_constants,
            pipeline: Self::rotation_pipeline(&device)?,
            bindgroup: rotations_bind_group,
        };

        let visibility = ComputePassResources {
            uniform: visibility_constants_buffer,
            dispatches: visibility_dispatches,
            constants: visibility_constants,
            pipeline: Self::visibility_pipeline(&device)?,
            bindgroup: visibility_bind_group,
        };

        let gpu = Self {
            device,
            queue,
            constants,
            rotation,
            visibility,
            output_buffers,
            output_surfaces_size,
            output_rings_size,
            output_longest_lines_size,
        };

        tracing::trace!("GPU pipline ready.");
        Ok(gpu)
    }

    /// We first initialize an wgpu `Instance`, which contains any "global" state wgpu needs.
    ///
    /// This is what loads the vulkan/dx12/metal/opengl libraries.
    fn instance() -> wgpu::Instance {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION,
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        })
    }

    /// Get the limits for the GPU.
    fn limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
        let limits = adapter.limits();
        // See: https://github.com/gfx-rs/wgpu/issues/8105
        let max_buffer_size = 0x7F_FFF_FFF;
        tracing::debug!("GPU limits: {limits:?}");
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 6,
            max_storage_buffer_binding_size: limits
                .max_storage_buffer_binding_size
                .min(max_buffer_size),
            max_buffer_size: limits.max_buffer_size.min(u64::from(max_buffer_size)),
            max_compute_workgroups_per_dimension: 1024,
            ..wgpu::Limits::default()
        }
    }

    /// The DEM rotation pipeline setup.
    fn rotation_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline> {
        // TODO: embed this if we ever make a proper binary release.
        //
        // Safety:
        //   We create our SPIRV with `cargo-gpu` that uses features that `wgpu`'s `naga` validator
        //   can't compile to `.wgsl`.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                entry_point: "rotate".into(),
                label: None,
                spirv: Some(wgpu::util::make_spirv_raw(include_bytes!(
                    "../../kernels/vulkan-and-cpu/kernel.spv"
                ))),
                ..Default::default()
            })
        };

        // The pipeline layout describes the bind groups that a pipeline expects
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&Self::create_rotation_bind_group_layout(device)?],
            push_constant_ranges: &[],
        });

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("rotate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        )
    }

    /// The total viewsheds pipeline setup.
    fn visibility_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline> {
        // TODO: embed this if we ever make a proper binary release.
        //
        // Safety:
        //   We create our SPIRV with `cargo-gpu` that uses features that `wgpu`'s `naga` validator
        //   can't compile to `.wgsl`.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                entry_point: "visibility".into(),
                label: None,
                spirv: Some(wgpu::util::make_spirv_raw(include_bytes!(
                    "../../kernels/vulkan-and-cpu/kernel.spv"
                ))),
                ..Default::default()
            })
        };

        // The pipeline layout describes the bind groups that a pipeline expects
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&Self::create_visibility_bind_group_layout(device)?],
            push_constant_ranges: &[],
        });

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("visibility"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        )
    }

    /// Setup the buffers for the rotation kernel.
    fn setup_rotation_buffers(
        device: &wgpu::Device,
        elevations: Vec<i16>,
        rotations_size: u64,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup)> {
        tracing::trace!("Creating GPU buffers for rotation kernel...");
        let input_constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rotation constants"),
            size: std::mem::size_of::<kernel::constants::Constants>().try_into()?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // All the elevation data.
        let input_elevations_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Elevations"),
                contents: bytemuck::cast_slice(&elevations),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // A buffer to read and write rotated elevation data to.
        let rw_rotation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rotated elevations"),
            size: rotations_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = Self::create_rotation_bind_group_layout(device)?;

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
                    resource: rw_rotation_buffer.as_entire_binding(),
                },
            ],
        });

        drop(elevations); // Free up RAM. Although it gets dropped anyway right??

        tracing::trace!("...rotation kernel GPU buffers created.");
        Ok((rw_rotation_buffer, input_constants_buffer, bind_group))
    }

    /// Setup the buffers.
    fn setup_visibility_buffers(
        device: &wgpu::Device,
        constants: kernel::constants::Constants,
        rotated_elevations_buffer: &wgpu::Buffer,
        output_rings_size: u64,
        longest_lines_size: u64,
    ) -> Result<(OutputBuffers, wgpu::Buffer, wgpu::BindGroup)> {
        tracing::trace!("Creating GPU buffers....");
        let input_constants_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visibility constants"),
            size: std::mem::size_of::<kernel::constants::Constants>().try_into()?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

        let output_longest_lines_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Longest lines data"),
            size: longest_lines_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let download_longest_linest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download longest lines data"),
            size: longest_lines_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = Self::create_visibility_bind_group_layout(device)?;
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
                    resource: rotated_elevations_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_surfaces_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_rings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_longest_lines_buffer.as_entire_binding(),
                },
            ],
        });

        let output_buffers = OutputBuffers {
            output_surfaces: output_surfaces_buffer,
            download_surfaces: download_surfaces_buffer,
            output_rings: output_rings_buffer,
            download_rings: download_rings_buffer,
            output_longest_lines: output_longest_lines_buffer,
            download_longest_lines: download_longest_linest_buffer,
        };

        tracing::trace!("...GPU buffers created.");
        Ok((output_buffers, input_constants_buffer, bind_group))
    }

    /// Bind group layout for the rotation kernel.
    fn create_rotation_bind_group_layout(device: &wgpu::Device) -> Result<wgpu::BindGroupLayout> {
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
                // Original elevations
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
                // Rotated elevations
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

    /// Bind group layout for the visibility kernel.
    fn create_visibility_bind_group_layout(device: &wgpu::Device) -> Result<wgpu::BindGroupLayout> {
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
                // Rotated elevations
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output: surface data
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        // This is the size of a single element in the buffer.
                        min_binding_size: std::num::NonZeroU64::new(4),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Output: longest lines of sight
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
            ],
        });

        Ok(layout)
    }

    /// Run all the compute stages/pipelines.
    pub fn run(
        &mut self,
        constants: kernel::constants::Constants,
    ) -> Result<(Vec<f32>, Vec<u32>, Vec<f32>)> {
        self.constants = constants;
        // The command encoder allows us to record commands that we will later submit to the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.compute_rotated_elevations(&mut encoder);
        self.compute_visibility(&mut encoder);

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
        let buffer_slice_surfaces = self.output_buffers.download_surfaces.slice(..);
        buffer_slice_surfaces.map_async(wgpu::MapMode::Read, |_| {});
        let buffer_slice_rings = self.output_buffers.download_rings.slice(..);
        buffer_slice_rings.map_async(wgpu::MapMode::Read, |_| {});
        let buffer_slice_longest_lines = self.output_buffers.download_longest_lines.slice(..);
        buffer_slice_longest_lines.map_async(wgpu::MapMode::Read, |_| {});

        // Wait for the GPU to finish working on the submitted work. This doesn't work on WebGPU, so we would need
        // to rely on the callback to know when the buffer is mapped.
        let result = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Err(error) = result {
            color_eyre::eyre::bail!("{error:?}");
        }

        // We can now read the data from the buffer.
        let ring_data = buffer_slice_rings.get_mapped_range();
        let surfaces_data = buffer_slice_surfaces.get_mapped_range();
        let longest_lines = buffer_slice_longest_lines.get_mapped_range();
        // Convert the data back to a slice of f32.
        let surfaces_result = bytemuck::cast_slice(&surfaces_data).to_vec();
        let ring_result = bytemuck::cast_slice(&ring_data).to_vec();
        let longest_lines_result = bytemuck::cast_slice(&longest_lines).to_vec();

        drop(surfaces_data);
        drop(ring_data);
        drop(longest_lines);

        self.output_buffers.download_surfaces.unmap();
        self.output_buffers.download_rings.unmap();
        self.output_buffers.download_longest_lines.unmap();

        Ok((surfaces_result, ring_result, longest_lines_result))
    }

    /// Compute a single sector.
    pub fn compute_rotated_elevations(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.rotation.constants.sine = self.constants.sine;
        self.rotation.constants.cosine = self.constants.cosine;
        self.queue.write_buffer(
            &self.rotation.uniform,
            0,
            bytemuck::bytes_of(&self.rotation.constants),
        );
        // pass, we cannot record to the encoder.
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.rotation.pipeline);
        compute_pass.set_bind_group(0, &self.rotation.bindgroup, &[]);
        compute_pass.dispatch_workgroups(
            self.rotation.dispatches.0,
            self.rotation.dispatches.1,
            self.rotation.dispatches.2,
        );
    }

    /// Compute visibility for a single sector.
    pub fn compute_visibility(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.visibility.constants.sine = self.constants.sine;
        self.visibility.constants.cosine = self.constants.cosine;
        self.queue.write_buffer(
            &self.visibility.uniform,
            0,
            bytemuck::bytes_of(&self.visibility.constants),
        );
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.visibility.pipeline);
        compute_pass.set_bind_group(0, &self.visibility.bindgroup, &[]);
        compute_pass.dispatch_workgroups(
            self.visibility.dispatches.0,
            self.visibility.dispatches.1,
            self.visibility.dispatches.2,
        );

        // Now we drop the compute pass, giving us access to the encoder again.
        drop(compute_pass);

        // We add a copy operation to the encoder. This will copy the data from the output buffer on the
        // GPU to the download buffer on the CPU.
        encoder.copy_buffer_to_buffer(
            &self.output_buffers.output_surfaces,
            0,
            &self.output_buffers.download_surfaces,
            0,
            self.output_surfaces_size,
        );

        encoder.copy_buffer_to_buffer(
            &self.output_buffers.output_rings,
            0,
            &self.output_buffers.download_rings,
            0,
            self.output_rings_size,
        );

        encoder.copy_buffer_to_buffer(
            &self.output_buffers.output_longest_lines,
            0,
            &self.output_buffers.download_longest_lines,
            0,
            self.output_longest_lines_size,
        );
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
