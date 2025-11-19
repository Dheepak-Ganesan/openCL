__kernel void eltwise_mul(__global float* a,__global float* b,__global float* c, int num)
{
    int  i = get_global_id(0);
    if(i < num)
        c[i] = a[i] * b[i];
}