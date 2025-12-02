__kernel void laplacian_filter(
    __global int* input,
    __global int* filter,
    __global int* output,
    const int kH,
    const int kW,
    const int outH,
    const int outW,
    const int height,
    const int width,
    const int max_abs_sum)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);

    if (grow >= outH || gcol >= outW) return;

    int sum = 0;
    for(int i=0; i<kH; i++){
        for(int j=0; j<kW; j++){
            sum += input[(grow+i)*width + (gcol+j)] * filter[i*kW + j];
        }
    }

    // Dynamic normalization: scale sum to [0,255] with 128 mid-gray
    float normalized = 128.0f + ((float)sum * 127.0f) / (float)max_abs_sum;
    int shifted = (int)round(normalized);

    if(shifted < 0) shifted = 0;
    if(shifted > 255) shifted = 255;

    output[grow*outW + gcol] = shifted;
}

/*
Laplacian Normalization:
The Laplacian filter detects edges by computing second-order derivatives. Its output can be negative (dark edges), positive (bright edges), or zero (no edge). Since image display expects pixel values in 0–255, we apply normalization/shift:

output_pixel = sum / factor + 128

Negative sums → mapped <128 → darker pixels
Zero → mapped to 128 → mid-gray
Positive sums → mapped >128 → brighter pixels

This ensures all edge responses are visible in a standard grayscale image without clipping.
*/