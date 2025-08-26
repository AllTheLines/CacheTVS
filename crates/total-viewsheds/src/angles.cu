#include <cub/block/block_scan.cuh>

typedef struct {
    unsigned int angles;
    unsigned int total_bands;
    unsigned int max_los_as_points;
    unsigned int dem_width;
    unsigned int tvs_width;
    float observer_height;
} calculation_constants;

#define EARTH_RADIUS_SQUARED 12742000.0

#define TOTAL_BANDS 72000000
#define TVS_WIDTH 6000
#define MAX_LOS_POINTS 6000

#define BLOCK_SIZE 8

#define TAN_ONE_RAD (float)0.0174533

#define ull unsigned long long


extern "C" __global__ void angle_kernel(
    // Constants for the calculations.
    calculation_constants constants,
    // Every single DEM point's elevation.
    const float* __restrict__ elevations,
    // Every single DEM point's distance from the sector-orthogonal axis.
    const float* __restrict__ distances,
    // Deltas used to build a band. Deltas are always the same for both front and back bands.
    //
    // Deltas are simply the numerical difference between DEM IDs in a band of
    // sight. DEM IDs are certainly different for every point in a DEM, but
    // conveniently the _differences_ between DEM IDs are identical. With the only
    // caveat that back-facing bands have opposite magnitudes. It should be stressed
    // that this feature of band data is a huge benefit to the space-requirements
    // (and thus speed) of the algorithm.
    // TODO: Explore:
    //         * storing the band deltas in local memory.
    //         * compressing repeating delta values -- is simple addition faster than
    //           memory accesses?
    const unsigned int* __restrict__ delta_pos,
    const unsigned int* __restrict__ delta_neg,
    float* result
) {
    // line_num tells us what (kernel_id, angle) we are at
    ull line_num = blockIdx.x;


    ull tvs_id = line_num;
    ull angle = blockIdx.y;

    // determine whether we are forwards or backwards facing
    // TODO: this doesn't seem to "fall out" of the implementation
    ull half_total_bands = TOTAL_BANDS/2;

    bool forward = tvs_id > half_total_bands;
    if (forward) {
        tvs_id -= half_total_bands;
    }

    ull pov_x = (tvs_id % TVS_WIDTH) + MAX_LOS_POINTS;
    ull pov_y = (tvs_id / TVS_WIDTH) + MAX_LOS_POINTS;

    // get the dem id for our pov which is where we start our calculation
    ull pov_id = (pov_x * constants.dem_width) + pov_y;

    // calculate the height
    const float pov_elevation = elevations[pov_id] + constants.observer_height;

    int deltas[BLOCK_SIZE];
    float angle_buf[BLOCK_SIZE];
    float distance[BLOCK_SIZE];

    int delta_index_start = (angle*MAX_LOS_POINTS) + threadIdx.x*BLOCK_SIZE;

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE / 4; i++) {
        reinterpret_cast<int4*>(&deltas)[i] = reinterpret_cast<const int4*>(&delta_pos[delta_index_start])[i];
        reinterpret_cast<float4*>(&distance)[i] = reinterpret_cast<const float4*>(&distances[delta_index_start])[i];
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        int delta = forward ? deltas[i] : -deltas[i];

        ull dem_id = pov_id + delta;

        float elevation_delta = elevations[dem_id] - pov_elevation;
        angle_buf[i] = elevation_delta / distance[i];
    }

    float prefix_max[BLOCK_SIZE];

    using BlockScan = cub::BlockScan<float, MAX_LOS_POINTS / BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockScan(temp_storage)
        .InclusiveScan(angle_buf, prefix_max, cuda::maximum<>{});


    float sum = 0.0;

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (angle_buf[i] >= prefix_max[i]) {
            sum += distance[i] * TAN_ONE_RAD;
        }
    }


    if (sum > 0.0) {
        atomicAdd(&result[tvs_id], sum);
    }
}
