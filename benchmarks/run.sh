#!/bin/env bash

export RUST_BACKTRACE=1
export RUST_LOG=trace

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"

backend=$1

function geojson_area() {
	local file=$1
	local temp=/tmp/viewshed.json

	cp "$file" "$temp"

	output=$(
		ogrinfo \
			"$temp" \
			-dialect sqlite \
			-sql "SELECT SUM(ST_Area(ST_Transform(geometry, 6933))) AS area_m2 FROM viewshed"
	)

	area=$(echo "$output" | grep -oP 'area_m2 \(Real\) = \K[0-9.]+')
	echo "$area"
}

# Do the calculations
export RUSTFLAGS='-Ctarget-cpu=native'
time cargo run --features ring_data --release -- \
	compute "$PROJECT_ROOT/benchmarks/cardiff.bt" \
	--scale 100 \
	--rings-per-km 3 \
	--backend "$backend" \
	--process all \
	--thread-count 1

if [[ $backend == "vulkan" ]]; then
	# On Github Actions there's no real GPU so it uses a software GPU, which seems to give
	# very different results, so there's no point doing a diff on its benchmark viewshed.
	exit 0
fi

# Reconstruct a viewshed from the centre of the DEM
time cargo run --release -- \
	viewshed output \
	-- -3.1230,51.4898

ls -alh output/viewsheds

expected_area=$(geojson_area "benchmarks/cardiff-viewshed.json")
actual_area=$(geojson_area output/viewsheds/-3.122999906539917-51.48979949951172.json)

diff=$(echo "$actual_area - $expected_area" | bc -l | tr -d '-')
limit=$(echo "$actual_area * 0.01" | bc -l)

if (($(echo "$diff <= $limit" | bc -l))); then
	echo "Viewhsed area within 1% of existing benchmark"
else
	echo "Benchmark viewshed changed too much ${expected_area} vs ${actual_area}"
	exit 1
fi
