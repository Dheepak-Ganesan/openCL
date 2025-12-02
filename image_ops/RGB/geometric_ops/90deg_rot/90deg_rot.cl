__kernel void rotate_90_clkWise(__global int* input,__global int* output,const int height,const int width,const int channels)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int gdep = get_global_id(2);

    if (grow >= height || gcol >= width || gdep >= channels) return;
    int index = gdep * (height*width) + grow * width + gcol;

    /*
    while rotating 90deg, the new row becomes the old columns
    and the new column becomes Height-old_row-1 and (HxW becomes WxH)
    
    new row = [old_col], 
    new col = [height - old_row -1]

    new_width = height (because after rotation, the output shape is width x height)
    channel remains the same
    */

    int rotate_90_clkWise_idx = gdep * (width * height) + gcol * height + ( height-grow-1);
    output[rotate_90_clkWise_idx] = input[index];
}
