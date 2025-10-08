
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

#define THREAD_COUNT 1000

#define BLOCK_SIZE (MAX_LOS_POINTS/THREAD_COUNT)


#define IMAGE_WIDTH 18000

#define TAN_ONE_RAD ((float)0.0174533)

__shared__ short line[MAX_LOS_POINTS*2];
__shared__ int line_idxs[MAX_LOS_POINTS];

#define ull unsigned long long

extern "C" __global__ void angle_kernel(
    const short* __restrict__ elevations,
    const int* __restrict__ idxs,
    float* __restrict__ result
) {
    // the input is (2*MAX_LOS_POINTS) wide, MAX_LOS_POINTS deep
    ull elevations_global = ((ull)blockIdx.y * (ull)MAX_LOS_POINTS * (ull)MAX_LOS_POINTS * 2ULL) + ((ull)blockIdx.x * (ull)MAX_LOS_POINTS * 2ULL);
    ull idxs_global = ((ull)blockIdx.y * (ull)MAX_LOS_POINTS * (ull)MAX_LOS_POINTS) + ((ull)blockIdx.x * (ull)MAX_LOS_POINTS);

    // GPUs like when adjacent threads load adjacent memory so that it can "coalesce" the read from global memory
    // GPUs also like it when you load at least 32 bits, so we make sure our access is 32-bit aligned and reinterpret_cast

    // load in the 2*MAX_LOS_POINTS (MAX_LOS_POINTS 32-bit integers) elevations for our particular line of sight
    for (int i = threadIdx.x; i < MAX_LOS_POINTS; i += THREAD_COUNT) {
        reinterpret_cast<int*>(&line)[i] = reinterpret_cast<const int*>(&elevations[elevations_global])[i];
    }

    // load in the index that we will store to since our data has been rotated
    for (ull i = threadIdx.x; i < MAX_LOS_POINTS; i += THREAD_COUNT) {
        line_idxs[i] = idxs[idxs_global + i];
    }

    // Make sure that all threads have access to the newly loaded shared memory
    __syncthreads();

    const int TILE_SIZE = 1;

    for (int tiled_off = 0; tiled_off < MAX_LOS_POINTS; tiled_off += THREAD_COUNT) {
        int pov = tiled_off + (threadIdx.x);

        // get the first height which will be our pov
        float pov_height = (float)line[pov];
        float max_angle = -2000.0;
        float sum = 0.0;

        // start the loop from the next elevation over as it is our first point to check
        for (int point = pov+1; point < pov+MAX_LOS_POINTS; point++) {
            float elevation_delta = ((float)line[point]) - pov_height;
            // the distance from the first point is our distance (since we're only ever going straight)
            // and each distance should be 100m
            float distance = (float)((point - pov)*100);
            float angle = (elevation_delta / distance) - (distance / EARTH_RADIUS_SQUARED);

            if (angle >= max_angle) {
                max_angle = angle;
                sum += distance * TAN_ONE_RAD;
            }
        }

        int result_idx = line_idxs[pov];
        if (result_idx > 0) {
            result[result_idx] = sum;
        }
    }

}
