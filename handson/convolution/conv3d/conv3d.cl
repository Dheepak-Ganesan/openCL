__kernel void conv3d(__global int* input,__global int* filter,__global int* output,
const int D, const int H,const int W, const int kD,const int kH,const int kW,const int oD,const int oH,const int oW)
{
    //(x,y,z) - output dimensions
    int out_x = get_global_id(0); // output dim x (Depth)
    int out_y = get_global_id(1); // output dim y (height)
    int out_z = get_global_id(2); // output dim z (width)

    int sum = 0;

    // Triple nested loop for 3D convolution:
    /*
    for dz in 0..2
        for dy in 0..2
            for dx in 0..2
                sum += input[z+dz][y+dy][x+dx] * kernel[dz][dy][dx];
    */
    for(int i = 0 ; i < kD; i++)
    {
        for(int j = 0 ; j < kH ; j++)
        {
            for(int k = 0 ; k < kW ; k++)
            {
                int inp_value = input[( (out_x + i) * (H * W)) + (out_y + j) * W + (out_z + k)];
                int filter_value = filter[ (i * (kH * kW)) + j * kW + k];
                sum+=inp_value*filter_value;
            }
        }
    }

    if( out_x < oD && out_y < oH && out_z < oW)
    {
        int out_idx = out_x * (oH*oW) + out_y * oW + out_z;
        output[out_idx] = sum;
    }

}