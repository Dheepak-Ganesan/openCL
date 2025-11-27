__kernel void conv2d(
    __global int* input,
    __global int* k,
    __global int* output,
    const int kH,
    const int kW,
    const int outH,
    const int outW,
    const int height,
    const int width
)
{
    int h = get_global_id(0); // row
    int w = get_global_id(1); // column

    if (h >= outH || w >= outW)
        return;

    int sum = 0;

    for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++)
            sum += input[(h + kh) * width + (w + kw)] * k[kh * kW + kw];

    output[h * outW + w] = sum;
}
