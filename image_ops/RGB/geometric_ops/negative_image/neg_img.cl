__kernel void neg_img(__global int* input,const int height,const int width,const int channels)
{

    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int gdep = get_global_id(2);

    if(grow >= height || gcol >= width || gdep >= channels) return;
    int index = gdep * height * width + grow * width + gcol;
    input[index] = 255 - input[index];
}