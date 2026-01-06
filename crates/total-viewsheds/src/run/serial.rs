//! For kernels that run each angle in serial.

use color_eyre::Result;

impl super::compute::Compute<'_> {
    /// `run_sequential` runs a sequential GPU or CPU kernel
    pub fn run_sequential(&mut self) -> Result<()> {
        let mut sector_surfaces = if Self::is_process_surfaces(&self.config.process) {
            let blank = vec![0.0; usize::try_from(self.dem.computable_points_count)?];
            self.total_surfaces.clone_from(&blank);
            blank
        } else {
            Vec::new()
        };

        let mut longest_lines = if Self::is_process_longest_lines(&self.config.process) {
            self.longest_lines = vec![
                crate::los_pack::LineOfSightPacked::default();
                usize::try_from(self.dem.computable_points_count)?
            ];
            vec![0.0; usize::try_from(self.dem.computable_points_count)?]
        } else {
            Vec::new()
        };

        for angle in 0..super::compute::SECTOR_STEPS {
            let mut sector_ring_data = vec![0; self.total_reserved_rings];
            let trig = kernel::rotation::Rotator::calculate_trig(f32::from(angle));
            self.constants.sine = trig.0;
            self.constants.cosine = trig.1;
            self.compute_sector(
                angle,
                &mut sector_surfaces,
                &mut sector_ring_data,
                &mut longest_lines,
            )?;

            if Self::is_process_viewsheds(&self.config.process) {
                match &self.config.output_directory {
                    Some(_) => {
                        self.save_sector_ring_data(angle, &sector_ring_data)?;
                    }
                    None => self.ring_data.push(sector_ring_data.clone()),
                }
            }

            if Self::is_process_longest_lines(&self.config.process) {
                self.increment_longest_lines(&longest_lines, angle)?;
                if angle == super::compute::SECTOR_STEPS - 1 {
                    self.render_longest_lines()?;
                }
            }
        }

        if Self::is_process_surfaces(&self.config.process) {
            self.add_sector_surfaces_to_running_total(&sector_surfaces);
            self.render_total_surfaces()?;
        }

        Ok(())
    }

    /// Create a GPU-friendly bitmask of flags to use in the kernel.
    pub fn bitmask_flags_for_kernel(processes: &[crate::config::Process]) -> u32 {
        use kernel::constants as kernel;
        let mut flags = 0u32;
        for process in processes {
            match process {
                crate::config::Process::All => {
                    flags |= kernel::Flag::TotalSurfaces.bit() | kernel::Flag::RingData.bit();
                }
                crate::config::Process::TotalSurfaces => {
                    flags |= kernel::Flag::TotalSurfaces.bit();
                }
                crate::config::Process::Viewsheds => flags |= kernel::Flag::RingData.bit(),
                crate::config::Process::LongestLines => {
                    flags |= kernel::Flag::LongestLines.bit();
                }
            }
        }
        flags
    }

    /// Compute a single sector.
    pub fn compute_sector(
        &mut self,
        angle: u16,
        cumulative_surfaces: &mut [f32],
        ring_data: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        tracing::info!("Running kernel for {angle}°");
        match self.config.backend {
            crate::config::Backend::VulkanCPU => {
                self.compute_sector_cpu_vulkan(cumulative_surfaces, ring_data, longest_lines)?;
            }
            crate::config::Backend::Vulkan => {
                self.compute_sector_vulkan(cumulative_surfaces, ring_data, longest_lines)?;
            }
            #[expect(clippy::unimplemented, reason = "CPU kernel is only multithreaded")]
            crate::config::Backend::CPU => {
                unimplemented!();
            }

            #[expect(clippy::unimplemented, reason = "Coming Soon!")]
            crate::config::Backend::Cuda => unimplemented!(),
        }

        Ok(())
    }

    /// Do a whole sector calculation on the GPU using Vulkan.
    fn compute_sector_vulkan(
        &mut self,
        cumulative_surfaces: &mut [f32],
        rings: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        let Some(gpu) = self.vulkan.as_mut() else {
            color_eyre::eyre::bail!("`self.gpu` not instantiated yet.");
        };

        let (surfaces_data, rings_data, longest_lines_data) = gpu.run(self.constants)?;
        if Self::is_process_surfaces(&self.config.process) {
            cumulative_surfaces.copy_from_slice(surfaces_data.as_slice());
        }
        if Self::is_process_viewsheds(&self.config.process) {
            rings.copy_from_slice(rings_data.as_slice());
        }
        if Self::is_process_longest_lines(&self.config.process) {
            longest_lines.copy_from_slice(longest_lines_data.as_slice());
        }
        Ok(())
    }

    /// Do a whole sector calculation on the CPU.
    fn compute_sector_cpu_vulkan(
        &self,
        cumulative_surfaces: &mut [f32],
        ring_data: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        let chocolate_box_size = kernel::chocolate_box::size(self.dem.width, self.dem.tvs_width);
        let mut rotated_elevations = vec![0.0; usize::try_from(chocolate_box_size)?];
        for chocolate_id in 0..(self.dem.computable_points_count * 2) {
            let chocolate = kernel::chocolate_box::Rotator::new_from_cached_trig(
                chocolate_id,
                self.dem.width,
                self.dem.tvs_width,
                self.constants.sine,
                self.constants.cosine,
            );
            // Note that we _anti_ rotate because anti-rotating the DEM grid has the effect of normally
            // rotating the line of sight. Which is just more intuitive to work with when debugging.
            chocolate
                .anti_rotate_value_nearest_neighbour(&self.dem.elevations, &mut rotated_elevations);
        }

        let mut buffers = kernel::kernel::Buffers {
            constants: &self.constants,
            elevations: &rotated_elevations,
            cumulative_surfaces,
            longest_lines,
            ring_data,
        };

        for tvs_id in 0..self.constants.total_bands {
            kernel::kernel::Kernel::run(tvs_id, &mut buffers);
        }

        Ok(())
    }

    /// Add the accumulated total surface areas for the current sector to the running total.
    pub fn add_sector_surfaces_to_running_total(&mut self, cumulative_surfaces: &[f32]) {
        for (left, right) in self
            .total_surfaces
            .iter_mut()
            .zip(cumulative_surfaces.iter())
        {
            *left += right;
        }
    }

    /// Check to see if this angle increases the current longest line of sight for the point.
    pub fn increment_longest_lines(&mut self, longest_lines: &[f32], sector: u16) -> Result<()> {
        for (left, right) in self.longest_lines.iter_mut().zip(longest_lines.iter()) {
            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "Distances always fit in u32"
            )]
            let current = right.abs() as u32;
            if current > left.distance() {
                let angle = if *right >= 0.0 { sector } else { sector + 180 };
                let packed = crate::los_pack::LineOfSightPacked::new(current, angle)?;
                *left = packed;
            }
        }

        Ok(())
    }
}
