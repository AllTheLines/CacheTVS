#define M_PI ((float) 3.14159265358979323846)
#define ull unsigned long long
#define ll long long int


__device__ float to_radians(float degrees) {
    return (degrees/(float)180.0) * M_PI;
}

__device__ ll clamp_index(ll value, ll start, ll end) {
    ll bottom = value < start ? start : value;
    return bottom > end ? end : bottom;
}

extern "C" __global__ void rotate_kernel(
    const short* __restrict__ elevations,
    short* __restrict__ elevations_out,
    int* __restrict__ index_out,
    int angle_off
) {
    // This kernel is 1 x 2, so we can make use of max_los to
    int max_los = (gridDim.y*blockDim.y) / 2;
    int width = max_los * 3;

    int relative_x = (blockDim.x*blockIdx.x)+threadIdx.x;
    int relative_y = (blockDim.y*blockIdx.y)+threadIdx.y;

    // if our relative_x is higher than max_los we bail. This happens
    // when max_los is not divisible by 32
    // TODO: what happens when
    if (relative_x >= max_los) {
        return;
    }

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

    // TODO: figure out a good interpolation
    // This clamps both indexes between (0, width-1) to make sure that rotated
    // index is in bounds
    ull x_rot = clamp_index((ll)round(x_cos - y_sin) + y_center, 0, width-1);
    ull y_rot = clamp_index((ll)round(y_cos + x_sin) + x_center, 0, width-1);

    ull elevations_off = blockIdx.z * (max_los * (2 * max_los));
    ull elevation_index = (relative_x*(2*max_los)) + relative_y;
    ll rot_index = x_rot*width+y_rot;

    elevations_out[elevations_off+elevation_index] = elevations[rot_index];

    if (relative_y < max_los) {
        ull elevations_off = blockIdx.z * (max_los * max_los);
        ull elevation_index = (relative_x*max_los) + relative_y;

        bool in_bounds = (x_rot >= max_los && x_rot < max_los*2)
            && (y_rot >= max_los && y_rot < max_los*2);

        int tvs_idx = (x_rot-max_los)*max_los + (y_rot-max_los);

        index_out[elevations_off+elevation_index] = in_bounds ? tvs_idx : -1;
    }
}