__kernel void priv_matmul(__global int* a,__global int* b,__global int*c,const int width,const int height)
{
    int grow = get_global_id(0);
    int gcol = get_global_id(1);
    int sum = 0;

    for(int k = 0 ; k < width;k++)
    {
        int a_one = a[grow * width + k];
        int b_one = b[k*height + gcol];
        sum+= a_one * b_one;
    }

    int c_idx = grow * height + gcol;
    c[c_idx] = sum;

}