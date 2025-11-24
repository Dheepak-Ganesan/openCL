#define TILE_SIZE 2

__kernel void matmul_local(
    __global int* A,
    __global int* B,
    __global int* C,
    const unsigned int M,
    const unsigned int K,
    const unsigned int N)
{

    int row = get_global_id(0);
    int col = get_global_id(1);

    int lrow = get_local_id(0);
    int lcol = get_local_id(1);

    __local int Asub[TILE_SIZE][TILE_SIZE];
    __local int Bsub[TILE_SIZE][TILE_SIZE];

    int value = 0;

    int numTiles = K / TILE_SIZE;

   for(int tile = 0; tile < numTiles ; tile++)
   {
        Asub[lrow][lcol] = A[row * K + (tile * TILE_SIZE) + lcol];
        Bsub[lrow][lcol] = B[(lrow + tile * TILE_SIZE) * N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0 ; k < TILE_SIZE; k ++)
        {
            value+=Asub[lrow][k] * Bsub[k][lcol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

   }
    C[row * N + col] = value;
}
