__kernel void eltwise_tanh(__global float* a)
{
    int id = get_global_id(0);
    float x = a[id];
    float xneg = -x;

    a[id] = (exp(x) - exp(xneg)) / (exp(x) + exp(xneg));
}
