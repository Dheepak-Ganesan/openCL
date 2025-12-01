__kernel void brighten(__global int* input_image,const int width)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int index = gx * width + gy;
    input_image[index] = ( input_image[index] + 30) < 255 ? input_image[index] + 30 : 255;
}