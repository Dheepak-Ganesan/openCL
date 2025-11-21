__kernel void eltwise_sigmoid(__global float* a)
{
    // 1 / (1 + e^(-a))
    int id = get_global_id(0);

    float x = a[id]; // private mem usage
    a[id] = 1.0 / (1.0 + exp(-x));
}