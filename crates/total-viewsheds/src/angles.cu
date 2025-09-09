
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
#define OFFSET ((IMAGE_WIDTH-MAX_LOS_POINTS)/2)

#define TAN_ONE_RAD ((float)0.0174533)

__shared__ short line[IMAGE_WIDTH];

extern "C" __global__ void angle_kernel(
    // Every single DEM point's elevation.
    const short* __restrict__ elevations,
    float* __restrict__ result
) {
    int base_global = (blockIdx.x * IMAGE_WIDTH) + OFFSET;

    for (int i = threadIdx.x; i < IMAGE_WIDTH/2; i += (IMAGE_WIDTH/THREAD_COUNT/2)) {
        reinterpret_cast<int*>(&line)[i] = reinterpret_cast<const int*>(&elevations[base_global])[i];
    }


    const int TILE_SIZE = 2;

    int LINE_START = MAX_LOS_POINTS;

    for (int tiled_off = 0; tiled_off < MAX_LOS_POINTS; tiled_off += TILE_SIZE*THREAD_COUNT) {
        int thread_start = tiled_off + (threadIdx.x*TILE_SIZE);

        for (int pov = thread_start; pov < thread_start+TILE_SIZE; pov++) {
            float pov_height = (float)line[LINE_START+pov];
            float max_angle = -2000.0;
            float sum = 0.0;

            for (int point = pov+1; point < pov+MAX_LOS_POINTS; point++) {
                float elevation_delta = ((float)line[LINE_START+point]) - pov_height;
                float distance = (float)((point - pov)*100);
                float angle = elevation_delta / distance;
                if (angle >= max_angle) {
                    max_angle = angle;
                    sum += distance * TAN_ONE_RAD;
                }
            }

            result[blockIdx.x*MAX_LOS_POINTS+pov] = sum;
        }
    }

}
