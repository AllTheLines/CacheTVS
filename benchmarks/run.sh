#!/bin/env bash

set -e

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

sqlite_db_path="./output/benchmark.db"
rm $sqlite_db_path || true

# Do the calculations
time cargo run --features ring_data --release -- \
	compute "$PROJECT_ROOT/benchmarks/cardiff.tiff" \
	--rings-per-km 3 \
	--backend "$backend" \
	--process all \
	--viewsheds-db-path $sqlite_db_path \
	--thread-count 1

if [[ $backend == "vulkan" ]]; then
	# On Github Actions there's no real GPU so it uses a software GPU, which seems to give
	# very different results, so there's no point doing a diff on its benchmark viewshed.
	exit 0
fi

viewshed_file="output/viewsheds/-3.122999906539917-51.48979949951172.json"
rm $viewshed_file || true

# Reconstruct a viewshed from the centre of the DEM
if [[ $backend == "cpu" ]]; then
	time cargo run --release -- \
		viewshed $sqlite_db_path ./output \
		-- -3.1230,51.4898
else
	time cargo run --release -- \
		viewshed ./output ./output \
		-- -3.1230,51.4898
fi

ls -alh output/viewsheds

expected_area=$(geojson_area "benchmarks/cardiff-viewshed.json")
actual_area=$(geojson_area "$viewshed_file")

diff=$(echo "$actual_area - $expected_area" | bc -l | tr -d '-')
limit=$(echo "$actual_area * 0.01" | bc -l)

if (($(echo "$diff <= $limit" | bc -l))); then
	echo "Viewhsed area within 1% of existing benchmark"
else
	echo "Benchmark viewshed changed too much ${expected_area} vs ${actual_area}"
	exit 1
fi
