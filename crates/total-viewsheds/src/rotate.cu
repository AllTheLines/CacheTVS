#define M_PI 3.14159265358979323846
#define ull unsigned long long
#define ll long long int


__device__ double to_radians(double degrees) {
    return (degrees/180.0) * M_PI;
}

__device__ ll clamp_index(ll value, ll start, ll end) {
    ll bottom = value < start ? start : value;
    return bottom > end ? end : bottom;
}

extern "C" __global__ void rotate_kernel(
    // elevations has gridDim.z number of (MAX_LOS_POINTS * 3)^2 buffers
    const short* __restrict__ elevations,
    short* __restrict__ elevations_out,
//     const int* __restrict__ index_out,
    int angle_off
) {
    int max_los = (gridDim.y*blockDim.y) / 2;
    int width = max_los * 3;

    int relative_x = (blockDim.x*blockIdx.x)+threadIdx.x;
    int relative_y = (blockDim.y*blockIdx.y)+threadIdx.y;

    if (relative_x >= max_los) {
        return;
    }

//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//         printf("(%d, %d): (%d, %d)\n", gridDim.y, blockIdx.y, relative_x, relative_y);
//     }

    ll x_center = width / 2;
    ll y_center = width / 2;

    int x = relative_x + max_los;
    int y = relative_y + max_los;

    float angle_sin = sinf(to_radians(angle_off+blockIdx.z));
    float angle_cos = cosf(to_radians(angle_off+blockIdx.z));

    float x_sin = (float)(x - x_center) * angle_sin;
    float x_cos = (float)(x - x_center) * angle_cos;

    float y_sin = (float)(y - y_center) * angle_sin;
    float y_cos = (float)(y - y_center) * angle_cos;

    ll x_rot = clamp_index((ll)round(x_cos - y_sin) + y_center, 0, width-1);
    ll y_rot = clamp_index((ll)round(y_cos + x_sin) + x_center, 0, width-1);

    elevations_out[relative_x*max_los*2+relative_y] = elevations[x_rot*width+y_rot];
}