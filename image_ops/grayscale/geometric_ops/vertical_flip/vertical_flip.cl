__kernel void vertical_flip(__global int* input,__global int* output,const int width,const int height)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int index = grow * width + gcol;


    /*
    vertical flip:

    input:      output:
    00 01 02    10 11 12
    10 11 12    00 01 02

    here the col value stays the same,where as the row value changes
    so, row = height - row -1;
    */

    int flip_row = height - grow - 1;
    int flip_idx = flip_row * width + gcol;
    output[flip_idx] = input[index];
    
}