#!/bin/env bash

export RUST_BACKTRACE=1
export RUST_LOG=trace

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"

# Do the calculations
time cargo run --release -- \
	compute "$PROJECT_ROOT/benchmarks/cardiff_50km_aeqd_100m.bt" \
	--scale 100 \
	--rings-per-km 3 \
	--backend vulkan \
	--process all

# Reconstruct a viewshed from the centre of the DEM
time cargo run --release -- \
	viewshed output \
	-- -3.1230,51.4898

ls -alh output/viewsheds

diff \
	"$PROJECT_ROOT/output/viewsheds/-3.122999906539917-51.48979949951172.json" \
	"$PROJECT_ROOT/benchmarks/cardiff.json" ||
	true
