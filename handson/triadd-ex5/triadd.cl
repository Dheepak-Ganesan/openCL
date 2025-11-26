__kernel void triadd(__global int* a,__global int* b, __global int* c,__global int* d)
{
    int gid = get_global_id(0);
    d[gid] = a[gid] + b[gid] + c[gid];
}