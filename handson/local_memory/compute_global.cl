__kernel void compute_global(__global int* a, __global int* b, __global int* res)
{
    int global_id = get_global_id(0);
    int global_size = get_global_size(0); 
    if(global_id + 1 < global_size)
        res[global_id] = a[global_id + 1] + b[global_id + 1];
    else
        res[global_id] = a[global_id] + b[global_id];
}