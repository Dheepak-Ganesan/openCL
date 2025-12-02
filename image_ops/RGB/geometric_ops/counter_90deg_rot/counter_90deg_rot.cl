__kernel void rotate_90_counterClkWise(__global int* input,__global int* output,const int height,const int width,const int channels)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int gdep = get_global_id(2);

    if(grow >= height || gcol >= width || gdep >= channels) return;
    int index = gdep * height * width + grow * width + gcol;

    /*
    while rotating 90deg counter clockwise, the new row becomes width - oldcol -1
    and the new column becomes old_row
    
    new row = [width - old_col - 1], 
    new col = [old_row]

    new_width = height (because after rotation, the output shape is width x height)
    channels remains the same
    */

    int rotate_90_counterClkWise_idx = gdep * width * height + (width - gcol - 1) * height + grow;
    output[rotate_90_counterClkWise_idx] = input[index];
}
