//! A custom matcher for matching coordinates.

#![cfg(test)]

use googletest::prelude::*;

#[derive(MatcherBase)]
/// A custom matcher for coordinates. Really only needed to match NaNs.
pub struct CoordMatcher {
    expected: glam::Vec2,
}

impl Matcher<glam::Vec2> for CoordMatcher {
    fn matches(&self, actual: glam::Vec2) -> googletest::matcher::MatcherResult {
        let x_nan = actual.x.is_nan() && self.expected.x.is_nan();
        let y_nan = actual.y.is_nan() && self.expected.y.is_nan();
        if x_nan && y_nan {
            return googletest::matcher::MatcherResult::Match;
        }

        if verify_float_eq!(actual.x, self.expected.x).is_err() {
            return googletest::matcher::MatcherResult::NoMatch;
        }

        if verify_float_eq!(actual.y, self.expected.y).is_err() {
            return googletest::matcher::MatcherResult::NoMatch;
        }

        googletest::matcher::MatcherResult::Match
    }

    fn describe(
        &self,
        _: googletest::matcher::MatcherResult,
    ) -> googletest::description::Description {
        format!("{:?}", self.expected).into()
    }
}

/// Custom matcher for coordinates.
#[must_use]
pub fn good_coordinate(expected: glam::Vec2) -> CoordMatcher {
    CoordMatcher { expected }
}
