__kernel void rotate_90_counterClkWise(__global int* input,__global int* output,const int height,const int width)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);

    int index = grow * width + gcol;

    /*
    while rotating 90deg counter clockwise, the new row becomes width - oldcol -1
    and the new column becomes old_row
    
    new row = [width - old_col - 1], 
    new col = [old_row]

    new_width = height (because after rotation, the output shape is width x height)
    */

    int rotate_90_counterClkWise_idx = (width - gcol - 1) * height + grow;
    output[rotate_90_counterClkWise_idx] = input[index];
}
