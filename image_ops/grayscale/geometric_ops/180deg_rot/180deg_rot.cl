__kernel void rotate_180_clkWise(__global int* input,__global int* output,const int height,const int width)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int index = grow * width + gcol;

    /*
    input       output
    00 01 02    22 21 20
    10 11 12    12 11 10
    20 21 22    02 01 00

    now the new_row = Height - 1 - old_row;
    new_col = Width - 1- old_col;
    */

    int new_row = height - 1 - grow;
    int new_col =  width - 1 - gcol;
    int rotate_180_clkWise_idx = new_row * width + new_col;
    output[rotate_180_clkWise_idx] = input[index];
}