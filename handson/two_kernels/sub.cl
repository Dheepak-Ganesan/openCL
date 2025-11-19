__kernel void eltwise_sub(__global int* a, __global int* b, __global int* c, __global int* d, __global int* res,const int count)
{
    int id = get_global_id(0);
    if(id < count)
         res[id] = (a[id] - b[id]) - (c[id] - d[id]);
}