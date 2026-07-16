//! Check that a polygon is valid. Only used for non-production code like debugging and tests.
//!
//! A lot of this code was originally copy-pasted from LLM chat. LLMs neither guarantee the license
//! nor the quality of code.

#![expect(clippy::indexing_slicing, reason = "This is not used in production")]

/// The tolerance that values must exceed in ordered to be considered for validity checks.
const TOLERANCE: f64 = 1e-6;

/// A single line from a polygon.
#[derive(Debug, Clone, Copy)]
struct LineSegment {
    /// The starting coordinate.
    from: crate::polygon::Coordinate,
    /// The ending coordinate.
    to: crate::polygon::Coordinate,
}

/// Check if a polygon is valid.
pub(crate) fn check(vertices: &[crate::vertices::Vertex]) -> bool {
    let mut lines = Vec::new();
    let mut previous: Option<crate::polygon::Coordinate> = None;

    for vertex in vertices {
        let current = vertex.coordinate;

        if let Some(prev) = previous {
            // Skip near-zero-length edges (e.g. End/Start pairs at the same point)
            if (current.x - prev.x).abs() < TOLERANCE && (current.y - prev.y).abs() < TOLERANCE {
                continue;
            }

            if are_points_close(prev, current) {
                continue;
            }
            lines.push(LineSegment {
                from: prev,
                to: current,
            });
        }

        previous = Some(current);
    }
    let total = lines.len();

    for i in 0..total {
        for j in (i + 1)..total {
            // Skip adjacent lines because they naturally intersect at their shared vertex
            if j == i + 1 || (i == 0 && j == total - 1) {
                continue;
            }

            if do_lines_intersect(lines[i], lines[j]) {
                tracing::error!(
                    "Self-intersection at {i}/{j} ({total}): {:?}, {:?}",
                    lines[i],
                    lines[j]
                );
                return true;
            }
        }
    }
    false
}

/// Do the bounding boxes of 2 lines overlap?
fn bounding_boxes_overlap(l1: LineSegment, l2: LineSegment) -> bool {
    let l1_min_x = l1.from.x.min(l1.to.x);
    let l1_max_x = l1.from.x.max(l1.to.x);
    let l1_min_y = l1.from.y.min(l1.to.y);
    let l1_max_y = l1.from.y.max(l1.to.y);

    let l2_min_x = l2.from.x.min(l2.to.x);
    let l2_max_x = l2.from.x.max(l2.to.x);
    let l2_min_y = l2.from.y.min(l2.to.y);
    let l2_max_y = l2.from.y.max(l2.to.y);

    l1_min_x <= l2_max_x && l2_min_x <= l1_max_x && l1_min_y <= l2_max_y && l2_min_y <= l1_max_y
}

/// Are 2 points considered so "close" that, for the purposes of polygon validity, they're basically
/// the same point?
fn are_points_close(left: crate::polygon::Coordinate, right: crate::polygon::Coordinate) -> bool {
    let scale = left
        .x
        .abs()
        .max(left.y.abs())
        .max(right.x.abs())
        .max(right.y.abs())
        .max(1.0);
    let tolerance = scale * TOLERANCE;
    (left.x - right.x).abs() < tolerance && (left.y - right.y).abs() < tolerance
}

/// Find the orientation of an ordered triplet of points.
///
/// Returns:
///  0 --> All are collinear
///  1 --> Clockwise
///  2 --> Counterclockwise
fn orientation(
    first: crate::polygon::Coordinate,
    second: crate::polygon::Coordinate,
    third: crate::polygon::Coordinate,
) -> i32 {
    let value = (second.x - first.x).mul_add(
        -(third.y - second.y),
        (second.y - first.y) * (third.x - second.x),
    );

    // Using a tiny epsilon value to account for floating-point inaccuracies
    if value.abs() < 1e-4f64 {
        return 0;
    }

    if value > 0.0 { 1 } else { 2 }
}

/// Is `second` strictly inside the segment `first` to `third`?
fn is_on_segment(
    first: crate::polygon::Coordinate,
    second: crate::polygon::Coordinate,
    third: crate::polygon::Coordinate,
) -> bool {
    let dist_to_first = (second.x - first.x).hypot(second.y - first.y);
    let dist_to_third = (second.x - third.x).hypot(second.y - third.y);
    let scale = first
        .x
        .abs()
        .max(first.y.abs())
        .max(third.x.abs())
        .max(third.y.abs())
        .max(1.0);
    let tolerance = scale * TOLERANCE;
    if dist_to_first < tolerance || dist_to_third < tolerance {
        return false;
    }
    second.x <= first.x.max(third.x)
        && second.x >= first.x.min(third.x)
        && second.y <= first.y.max(third.y)
        && second.y >= first.y.min(third.y)
}

/// Do 2 lines intersect?
fn do_lines_intersect(l1: LineSegment, l2: LineSegment) -> bool {
    if !bounding_boxes_overlap(l1, l2) {
        return false;
    }

    // If any endpoint of one segment is within tolerance of any endpoint of the other,
    // treat this as a touch, not an intersection.
    for i in [l1.from, l1.to] {
        for j in [l2.from, l2.to] {
            if are_points_close(i, j) {
                return false;
            }
        }
    }

    let p1 = l1.from;
    let q1 = l1.to;
    let p2 = l2.from;
    let q2 = l2.to;

    // Find the four orientations needed for general and special cases
    let orientation1 = orientation(p1, q1, p2);
    let orientation2 = orientation(p1, q1, q2);
    let orientation3 = orientation(p2, q2, p1);
    let orientation4 = orientation(p2, q2, q1);

    // General Case: The segments cross each other
    if orientation1 != orientation2 && orientation3 != orientation4 {
        return true;
    }

    // Special Cases: The segments are collinear and touch/overlap
    if orientation1 == 0i32 && is_on_segment(p1, p2, q1) {
        return true;
    }
    if orientation2 == 0i32 && is_on_segment(p1, q2, q1) {
        return true;
    }
    if orientation3 == 0i32 && is_on_segment(p2, p1, q2) {
        return true;
    }
    if orientation4 == 0i32 && is_on_segment(p2, q1, q2) {
        return true;
    }

    false
}
