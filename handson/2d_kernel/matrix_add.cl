__kernel void matrix_add(__global int* A,
                         __global int* B,
                         __global int* C,
                         const int width)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    int idx = row * width + col;

    C[idx] = A[idx] + B[idx];
}
