#define size 5

__kernel void conv2d_local(__global int* input_dev, __global int* kernel_dev, 
                           __global int* out_dev, const int KH, const int KW, 
                           const int outH, const int outW, const int HEIGHT, 
                           const int WIDTH, const int WG, const int patch)
{
    int out_row = get_global_id(0);
    int out_col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    int grp_x = get_group_id(0);
    int grp_y = get_group_id(1);

    __local int localmem[size][size];

    // Load tile with halo
    for(int i = local_row; i < patch; i += WG)
    {
        for(int j = local_col; j < patch; j += WG)
        {
            int gx = grp_x * WG + i;
            int gy = grp_y * WG + j;
            
            if (gx < HEIGHT && gy < WIDTH)
                localmem[i][j] = input_dev[gx * WIDTH + gy];
            else
                localmem[i][j] = 0;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute convolution 
    // since the global dimensions are actual size, while storing it in the output,there should be a boundary check..
    //..in order to properly store the output values with size input-kernel+1,instead of entire input size(in this case 6x6)
    if (out_row < outH && out_col < outW)
    {
        int sum = 0;
        for(int kx = 0; kx < KH; kx++)
        {
            for(int ky = 0; ky < KW; ky++)
            {
                sum += localmem[local_row + kx][local_col + ky] * 
                       kernel_dev[kx * KW + ky];
            }
        }
        
        out_dev[out_row * outW + out_col] = sum;
    }
}