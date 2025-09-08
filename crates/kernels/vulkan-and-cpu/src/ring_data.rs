//! Ring data is the raw data needed to reconstruct viewsheds.
//!
//! It is not needed for calculating total surface areas or longest lines of sight.

/// Helper to store ring data.
pub struct RingData<'ring_data> {
    /// The ring data buffer.
    pub ring_data: &'ring_data mut [u32],
    /// The amount of reserved space in the global ring data buffer.
    pub reserved_rings_per_band: u32,
    /// Where this point's ring data starts in the ring data buffer.
    pub start: usize,
    /// A cursor to keep tract of where we are in our little section of the buffer.
    pub index: usize,
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "
      `usize` values are only ever generated from `u32` values. So they can't truncate.
    "
)]
impl RingData<'_> {
    /// Save ring data.
    pub fn save(&mut self, value: usize) {
        if self.index >= self.reserved_rings_per_band as usize {
            return;
        }
        self.ring_data[self.start + self.index] = value as u32;
        self.index += 1;
    }

    /// Make a note at the start of the ring sector data of how many rings we found.
    pub fn finish(&mut self) {
        self.ring_data[self.start] = self.index as u32;
    }
}
