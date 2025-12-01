__kernel void neg_img(__global int* input,const int width)
{

    int grow = get_global_id(0);
    int gcol = get_global_id(1);

    int index = grow * width + gcol;
    input[index] = 255 - input[index];
}