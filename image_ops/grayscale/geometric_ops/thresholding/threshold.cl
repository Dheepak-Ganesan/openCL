__kernel void threshold(__global int* input,const int width)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int index = gx * width + gy;
    input[index] = input[index] > 127 ? input[index] : 0;
}