__kernel void vertical_flip(__global int* input,__global int* output,const int width,const int height)
{
    int gH = get_global_id(0);
    int gW = get_global_id(1);
    int gD = get_global_id(2);
    int index = gD * (height * width) + gH * width + gW;

    /*
    Vertical flip (CHW layout):

    Example for channel 0 (Red):

    Input:       Output after flip:
    00 01 02     10 11 12
    10 11 12     00 01 02

    Columns remain the same.
    Rows are flipped: row = height - row - 1.
    This applies to all channels; channel order remains unchanged.
    */


    int flip_row = height - gH - 1;
    int flip_idx = gD * (height * width) + flip_row * width + gW;
    output[flip_idx] = input[index];
    
}