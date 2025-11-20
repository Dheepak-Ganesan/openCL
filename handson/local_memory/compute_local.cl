__kernel void compute_local(__global int* a, __global int* b , __global int* res, __local int* buffer1,__local int* buffer2)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    buffer1[local_id] = a[global_id];
    buffer2[local_id] = b[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(local_id + 1 < local_size)
        res[global_id] = buffer1[local_id  + 1] + buffer2[local_id + 1];  
    else
        res[global_id] = buffer1[local_id] + buffer2[local_id];

}