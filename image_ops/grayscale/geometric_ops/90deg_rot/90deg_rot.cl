__kernel void rotate_90_clkWise(__global int* input,__global int* output,const int height,const int width)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);

    int index = grow * width + gcol;

    /*
    while rotating 90deg, the new row becomes the old columns
    and the new column becomes Height-old_row-1
    
    new row = [old_col], 
    new col = [height - old_row -1]

    new_width = height (because after rotation, the output shape is width x height)
    */

    int rotate_90_clkWise_idx = gcol * height + ( height-grow-1);
    output[rotate_90_clkWise_idx] = input[index];
}
