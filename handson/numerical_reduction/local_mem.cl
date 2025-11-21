__kernel void compute_local(__global float* a,
                            __global float* b,
                            __global float* res,
                            __local float* localmem)
{
    int global_id = get_global_id(0);
    int local_id  = get_local_id(0);
    int local_size = get_local_size(0);

    // Step 1: Load data from global memory into local memory
    localmem[local_id] = a[global_id] + b[global_id];

    // Step 2: Synchronize work-items
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Reduction using stride halving
    for(int stride = local_size / 2; stride > 0; stride /= 2)
    {
        if(local_id < stride)
        {
            localmem[local_id] += localmem[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE); // synchronize after each step
    }

    // Step 4: Only local_id 0 writes the result for this work-group
    if(local_id == 0)
    {
        res[get_group_id(0)] = localmem[0];
    }
}
