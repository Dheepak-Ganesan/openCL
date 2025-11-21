__kernel void eltwise_relu6(__global float* a)
{
    int id = get_global_id(0);
    float max = a[id] > 0 ? a[id] : 0;
    a[id] = max > 6 ? 6 : max;
}

