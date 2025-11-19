__kernel void compute(__global const int* inA,
                      __global const int* inB,
                      __global int* out,
                      int offset)
{
    int gid = get_global_id(0);
    out[gid + offset] = inA[gid + offset] + inB[gid + offset];
}
