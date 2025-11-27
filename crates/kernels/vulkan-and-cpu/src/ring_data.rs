//! Ring data is the raw data needed to reconstruct viewsheds.
//!
//! It is not needed for calculating total surface areas or longest lines of sight.

/// Helper to store ring data.
pub struct RingData {
    /// The amount of reserved space in the global ring data buffer.
    pub reserved_rings_per_band: u32,
    /// Where this point's ring data starts in the ring data buffer.
    pub start: usize,
    /// A cursor to keep track of where we are in our little section of the buffer.
    pub cursor: usize,
}

impl RingData {
    /// Instantiate.
    pub const fn new(kernel_id: u32, reserved_rings_per_band: u32) -> Self {
        #[expect(
            clippy::as_conversions,
            reason = "This needs to run on the GPU where fallibility isn't possible"
        )]
        let start = (kernel_id * reserved_rings_per_band) as usize;
        // Reserve 0-index for the total count.
        let cursor = 1;

        Self {
            reserved_rings_per_band,
            start,
            cursor,
        }
    }

    /// Save ring data.
    pub fn save(&mut self, ring_data: &mut [u32], value: u32) {
        #[expect(
            clippy::as_conversions,
            reason = "
              `usize` values are only ever generated from `u32` values. So they can't truncate.
            "
        )]
        if self.cursor >= self.reserved_rings_per_band as usize {
            return;
        }
        ring_data[self.start + self.cursor] = value;
        self.cursor += 1;
    }

    /// Make a note at the start of the ring sector data of how many rings we found.
    pub fn finish(&self, ring_data: &mut [u32]) {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "
              `usize` values are only ever generated from `u32` values. So they can't truncate.
            "
        )]
        {
            ring_data[self.start] = self.cursor as u32;
        }
    }
}
