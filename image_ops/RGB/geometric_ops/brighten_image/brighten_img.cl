__kernel void brighten(__global int* input,
                        const int width,
                        const int height,
                        const int channels)
{
    int gH = get_global_id(0);  // height
    int gW = get_global_id(1);  // width
    int gC = get_global_id(2);  // channel

    if (gW >= width || gH >= height || gC >= channels) return;

    int index = gC * (height * width) + gH * width + gW; 

    input[index] = input[index] + 30;
}
