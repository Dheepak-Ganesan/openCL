__kernel void max_compute(__global int* a,__global int* res, __local int* localmem,const int height, const int width, const int depth)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int gdep = get_global_id(2);

    int lrow = get_local_id(0);
    int lcol = get_local_id(1);
    int ldep = get_local_id(2);

    int lheight = get_local_size(0);
    int lwidth = get_local_size(1);
    int ldepth = get_local_size(2);
    int local_size = lheight * lwidth * ldepth;

    int gid = (gdep * height * width) + grow * width + gcol;
    int lid = (ldep * lheight * lwidth) + lrow * lwidth + lcol;

    localmem[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = local_size/2 ; stride > 0 ; stride/=2)
    {
        if(lid < stride)
        {
            localmem[lid] = max(localmem[lid],localmem[lid + stride]);
        }
         barrier(CLK_LOCAL_MEM_FENCE);
    }

    int gx = get_group_id(0);
    int gy = get_group_id(1);
    int gz = get_group_id(2);

    int group_id = gz * (get_num_groups(0) * get_num_groups(1)) + gx *  get_num_groups(1) + gy;

    if(lid == 0 )
    {
        res[group_id] = localmem[0];
    }


}