__kernel void min_compute(
    __global int* input,
    __global int* min_res,
    __local int* localmem_min
)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    localmem_min[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = local_size / 2; stride > 0; stride /= 2)
    {
        if(lid < stride)
        {
            localmem_min[lid] = min(localmem_min[lid], localmem_min[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        int group_id = get_group_id(0);
        min_res[group_id] = localmem_min[0];
    }
}

__kernel void max_compute(
    __global int* input,
    __global int* max_res,
    __local int* localmem_max
)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    localmem_max[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = local_size / 2; stride > 0; stride /= 2)
    {
        if(lid < stride)
        {
            localmem_max[lid] = max(localmem_max[lid], localmem_max[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        int group_id = get_group_id(0);
        max_res[group_id] = localmem_max[0];
    }
}
