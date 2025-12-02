__kernel void horizontal_flip(__global int* input,
                              __global int* output,
                              const int width,
                              const int height)
{
    int gH = get_global_id(0);  // row
    int gW = get_global_id(1);  // column
    int gD = get_global_id(2);  // channel

    // compute input index in CHW layout
    int index = gD * (height * width) + gH * width + gW;

    /*
    Horizontal flip (CHW layout):

    Example for channel 0 (Red):

    Input:       Output after flip:
    00 01 02     02 01 00
    10 11 12     12 11 10

    Rows remain the same.
    Columns are flipped: col = width - col - 1.
    This applies to all channels; channel order remains unchanged.
    */

    int flip_col = width - gW - 1;
    int flip_idx = gD * (height * width) + gH * width + flip_col;

    output[flip_idx] = input[index];
}
