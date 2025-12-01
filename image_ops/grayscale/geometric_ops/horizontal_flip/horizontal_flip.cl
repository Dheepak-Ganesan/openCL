__kernel void horizontal_flip(__global int* input,__global int* output,const int width)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int index = grow * width + gcol;

    int flip_idx = grow * width + (width - gcol - 1);
    output[flip_idx] = input[index];
}