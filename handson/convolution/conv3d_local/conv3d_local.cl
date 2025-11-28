#define size 8

__kernel void conv3d_local(__global int* input,__global int* filter,__global int* output,
const int DEPTH, const int HEIGHT,const int WIDTH, const int kD,const int kH,const int kW,const int oD,const int oH,const int oW,const int WG,const int patch)
{
    //(x,y,z) - output dimensions
    int out_x = get_global_id(0); // output dim x (Depth)
    int out_y = get_global_id(1); // output dim y (height)
    int out_z = get_global_id(2); // output dim z (width)

    //local size = 2x2x2
    int local_x = get_local_id(0); // local dim x(Depth)  0,1
    int local_y = get_local_id(1); // local dim y(height) 0,1
    int local_z = get_local_id(2); // local dim z(width)  0,1


    int grp_x = get_group_id(0); // 4/2 = 2 (0,1)
    int grp_y = get_group_id(1); // 4/2 = 2 (0,1)
    int grp_z = get_group_id(2); // 4/2 = 2 (0,1)

    //load into local memory
    __local int localmem[size][size][size];

    // Load a tile of input into local memory (shared among threads in this workgroup)
    // Each thread loads multiple elements in the tile using strided loops (i,j,k += WG)
    // 'grp_x * WG + i' maps the local tile index to the correct global input index
    // Bounds check ensures we don’t read outside the input; out-of-bounds elements are padded with 0

  for(int i = local_x ; i < patch ; i+=WG)
    {
        for(int j = local_y ; j < patch ; j+=WG)
        {
            for(int k = local_z; k < patch ; k+=WG)
            {
                    int gx = grp_x * WG + i;
                    int gy = grp_y * WG + j;
                    int gz = grp_z * WG + k;
                    if(gx < DEPTH && gy <  HEIGHT && gz < WIDTH)
                        localmem[i][j][k] = input[gx * (HEIGHT * WIDTH) + gy * WIDTH + gz];
                    else
                        localmem[i][j][k] = 0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
                int inp_value = localmem[local_x + i][local_y + j][local_z + k];
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