//! Support using this crate in the browser with WASM.

// #![cfg(target_arch = "wasm32")]

#[cfg(target_arch = "wasm32")]
use js_sys::Array;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// WASM only supports simple types, so this is a simple representation of a polygon.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct PlainPolygon {
    /// The exterior coordinates of the polygon. x/y components are flattened, so the first
    /// coordinate is found at [0,1].
    exterior: Vec<f64>,
    /// The interior coordinates of the polygon. To find the starting/ending indices of each
    /// hole see `hole_indices`. x/y components are flattened, so the first coordinate is
    /// found at [0,1].
    interiors: Vec<f64>,
    /// Indices of where each hole starts and ends in `interiors`.
    hole_indices: Vec<u32>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl PlainPolygon {
    /// Getter for `exterior`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn exterior(&self) -> Vec<f64> {
        self.exterior.clone()
    }

    /// Getter for `interiors`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn interiors(&self) -> Vec<f64> {
        self.interiors.clone()
    }

    /// Getter for `hole_indices`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn hole_indices(&self) -> Vec<u32> {
        self.hole_indices.clone()
    }
}

#[cfg(target_arch = "wasm32")]
impl PlainPolygon {
    /// Instantiate.
    #[inline]
    #[must_use]
    pub fn new(growable_polygon: &crate::growable_polygon::GrowablePolygon) -> Self {
        let mut exterior = Vec::with_capacity(growable_polygon.vertices.len() * 2);
        for vertex in &growable_polygon.vertices {
            exterior.push(vertex.coordinate.x);
            exterior.push(vertex.coordinate.y);
        }

        let mut interiors = Vec::new();
        let mut hole_indices = Vec::with_capacity(growable_polygon.holes.len());

        for hole in &growable_polygon.holes {
            hole_indices.push(
                u32::try_from(interiors.len()).expect("Couldn't cast `interiors.len()` to `u32`"),
            );

            for vertex in hole {
                interiors.push(vertex.coordinate.x);
                interiors.push(vertex.coordinate.y);
            }
        }

        Self {
            exterior,
            interiors,
            hole_indices,
        }
    }
}

/// Reconstruct a viewshed from raw polar segments.
///
/// # Panics
///   When reconstructing the viewshed fails.
#[expect(
    unreachable_pub,
    reason = "We won't it to be `pub` for WASM but not avaiable outside the crate"
)]
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[must_use]
pub fn reconstruct(js_data: &js_sys::Array, dem_scale: f32) -> js_sys::Array {
    console_error_panic_hook::set_once();
    let mut rust_data: Vec<Vec<crate::segment::Segment>> = Vec::new();

    for segments_for_angle in js_data.iter() {
        let inner_js_array: js_sys::Array = segments_for_angle.unchecked_into();
        let mut inner_rust_vec = Vec::new();

        for segment in inner_js_array.iter() {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::as_conversions,
                reason = "
                  We're only worried about the truncation here, but it's very unlikely.
                  Remember that this value is actually a bitpack of 2 `u16`s.
                "
            )]
            let bitpack = segment
                .as_f64()
                .expect("Expected valid JavaScript `number`") as u32;

            inner_rust_vec.push(crate::segment::Segment(bitpack));
        }
        rust_data.push(inner_rust_vec);
    }

    let growable_polygons = crate::joiner::Joiner::join(&rust_data, dem_scale);

    let js_array = Array::new();
    for growable_polygon in growable_polygons {
        let plain_polygon = PlainPolygon::new(&growable_polygon);
        js_array.push(&JsValue::from(plain_polygon));
    }
    js_array
}
