__kernel void eltwise_add(__global int* a, __global int* b, __global int *c, const int offset,const int n)
{
    int id = get_global_id(0);
    if(id < n)
        c[id + offset] = a[id + offset] + b[id + offset];
}

