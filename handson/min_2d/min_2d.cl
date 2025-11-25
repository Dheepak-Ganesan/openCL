__kernel void min_compute(
    __global int* input,
    __global int* min_res,
    __local  int* localmem,
    const int width
)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    int gid = row * width + col;

    int lrow = get_local_id(0);
    int lcol = get_local_id(1);
    int lheight = get_local_size(0);
    int lwidth  = get_local_size(1);
    int local_size = lheight * lwidth;
    int lid = lrow * lwidth + lcol;

    int gx = get_group_id(0);
    int gy = get_group_id(1);
    int num_grp_y = get_num_groups(1);
    int grpid = gx * num_grp_y + gy;

    localmem[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = local_size >> 1; stride > 0; stride >>= 1)
    {
        if(lid < stride)
            localmem[lid] = min(localmem[lid], localmem[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
        min_res[grpid] = localmem[0];
}
