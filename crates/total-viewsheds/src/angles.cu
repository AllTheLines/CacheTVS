#include <cub/block/block_scan.cuh>

typedef struct {
    unsigned int angles;
    unsigned int total_bands;
    unsigned int max_los_as_points;
    unsigned int dem_width;
    unsigned int tvs_width;
    float observer_height;
} calculation_constants;

#define EARTH_RADIUS_SQUARED (float)12742000.0

#define TVS_WIDTH 6000
#define MAX_LOS_POINTS 6000
#define THREAD_COUNT 750


#define BLOCK_SIZE (MAX_LOS_POINTS / THREAD_COUNT)

#define WIDTH (3 * MAX_LOS_POINTS)
#define IMAGE_WIDTH (WIDTH)

#define TAN_ONE_RAD ((float)0.0174533)

#define ull unsigned long long

__shared__ short line[WIDTH];

__device__ void compute_point(int points, float* result) {
    int pov_id = MAX_LOS_POINTS + points;
    float pov_height = (float)line[pov_id];

    float angle_buf[BLOCK_SIZE];
    float prefix_max[BLOCK_SIZE];

    int los_base = pov_id + (threadIdx.x * BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        angle_buf[i] = ((los_base+i) == pov_id) ? -2000.0 :
            ((float)line[los_base+i] - pov_height) / fabs((float)((pov_id-(los_base+i))*10));
    }

    __syncthreads();

    using BlockScan = cub::BlockScan<float, THREAD_COUNT>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockScan(temp_storage)
        .InclusiveScan(angle_buf, prefix_max, cuda::maximum<>{});

    __syncthreads();

    float sum = 0.0;
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (angle_buf[i] >= prefix_max[i]) {
            sum += fabs((float)((pov_id-(los_base+i))*10)) * TAN_ONE_RAD;
        }
    }

    if (sum > 0.0) {
        int tvs_id = (blockIdx.x*MAX_LOS_POINTS)+(pov_id - MAX_LOS_POINTS);
        atomicAdd(&result[tvs_id], sum);
    }
    __syncthreads();
}

extern "C" __global__ void angle_kernel(
    // Every single DEM point's elevation.
    const short* __restrict__ elevations,
    float* result
) {
    int base_global = (blockIdx.x * IMAGE_WIDTH) + (threadIdx.x*(WIDTH/THREAD_COUNT)) + MAX_LOS_POINTS;
    int base_local = threadIdx.x*(WIDTH/THREAD_COUNT);

    for (int i = 0; i < WIDTH/THREAD_COUNT; i++) {
        line[base_local+i] = elevations[base_global+i];
    }
    __syncthreads();

    for (int points = 0; points < 6000; points++) {
        compute_point(points, result);
    }


//     int shared_idx = threadIdx.x * (WIDTH/THREAD_COUNT);
//     int global_idx = (blockIdx.x * IMAGE_WIDTH) + shared_idx;
//
//     for (int i = 0; i < WIDTH/THREAD_COUNT; i++) {
//         line[shared_idx+i] = elevations[global_idx+i];
//     }
//     __syncthreads();
//
//     int start = MAX_LOS_POINTS + threadIdx.x*(TVS_WIDTH/THREAD_COUNT);
//     for (int point = start; point < (start + (TVS_WIDTH/THREAD_COUNT)); point++) {
//         float pov_height = (float)line[point];
//         float max_angle = -2000.0;
//
//         float sum = 0.0;
//         for (int los_point = 1; los_point < MAX_LOS_POINTS; los_point++) {
//             float los = (float)line[point+los_point];
//
//             float distance = ((float)los_point * 10.0);
//             float angle = fabs(pov_height - los) / distance;
//             if (angle >= max_angle) {
//                 max_angle = angle;
//                 sum += distance * TAN_ONE_RAD;
//             }
//         }
//
//         int pov_id = (gridDim.x*MAX_LOS_POINTS) + (MAX_LOS_POINTS-point);
//         if (sum > 0.0) {
//             atomicAdd(&result[pov_id], sum);
//         }
//         result[pov_id] = 10.0;
//     }
//
//     for (int point = start; point < (start + (TVS_WIDTH/THREAD_COUNT)); point++) {
//         float pov_height = (float)line[point];
//         float max_angle = -2000.0;
//
//         float sum = 0.0;
//         for (int los_point = -1; los_point > -MAX_LOS_POINTS; los_point--) {
//             float los = (float)line[point+los_point];
//             float distance = ((float)los_point * 10.0);
//             float angle = fabs(pov_height - los) / distance;
//             if (angle >= max_angle) {
//                 max_angle = angle;
//                 sum += distance * TAN_ONE_RAD;
//             }
//         }
//
//         int tvs_id = (gridDim.x*MAX_LOS_POINTS) + (MAX_LOS_POINTS-point);
//
//         if (sum > 0.0) {
//            atomicAdd(&result[tvs_id], sum);
//         }
//         result[tvs_id] = 10.0;
//     }
}
