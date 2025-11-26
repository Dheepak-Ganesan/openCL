__kernel void reduction(__global int* input, __global int* result,__local int* localmem,const int totalnums)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int sum = 0;
    int locsize = get_local_size(0);
    int grpid = get_group_id(0);
   // int iteration = totalnums/get_global_size(0);
    int iteration = (totalnums + get_global_size(0) - 1) / get_global_size(0);
    for(int i = 0 ; i < iteration; i++)
    {
        int idx = gid * iteration + i;
        if(idx < totalnums)
         sum += input[idx];

    }
    localmem[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int stride = locsize/2; stride > 0; stride /= 2) {
        if(lid < stride)
            localmem[lid] += localmem[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
}


    if(lid == 0 )
        result[grpid] = localmem[0];    
}