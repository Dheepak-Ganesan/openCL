__kernel void eltwise_relu(__global int* relu_array)
{
    int id  = get_global_id(0);
    relu_array[id] = relu_array[id] > 0 ? relu_array[id] : 0;
}