__kernel void eltwise_3d(
    __global float* data,
    int width,
    int height,
    int depth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if (x >= width || y >= height || z >= depth)
        return;

    int idx = x + y * width + z * width * height;

    data[idx] = data[idx] * 2.0f;   // simple update
}
