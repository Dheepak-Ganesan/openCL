__kernel void gaussian_filter(__global int* input,__global int* filter,__global int* output,const int kH,const int kW,
const int outH,const int outW,const int height,const int width,const int div)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int index = grow * width + gcol;

    int sum = 0;
    for(int i = 0 ; i < kH; i++)
    {
        for(int j = 0 ; j < kW ; j++)
        {
            sum+=input[(grow + i) * width + (gcol + j)] * filter[ i * kW + j];
        }
    }

    if(grow < outH && gcol < outW)
    {
        output[grow * outW + gcol] = sum/div;
    }
}