__kernel void matmul(
    __global const int* A,   // M x K
    __global const int* B,   // K x N
    __global int* C,         // M x N
    int M,
    int K,
    int N
)
{
    int row = get_global_id(0);  // row of C
    int col = get_global_id(1);  // column of C

    int sum = 0.0f;

    for (int k = 0; k < K; k++)
    {
        int a = A[row * K + k];
        int b = B[k * N + col];
        sum += a * b;
    }

    C[row * N + col] = sum;
}
