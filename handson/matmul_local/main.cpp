#define __ENABLE_CL_EXCEPTIONS
#include <iostream>
#include <vector>
#include "cl.hpp"
#include "util.hpp"
using namespace std;

#define M 4
#define K 4
#define N 4

int main()
{
    vector<cl::Platform> platforms_list;
    cl::Platform::get(&platforms_list);
    cl::Platform platform = platforms_list[0];

    vector<cl::Device> devices_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&devices_list);
    cl::Device device = devices_list[0];

    cl::Context context(device);
    cl::CommandQueue queue(context,device);

    vector<int> a(M * K);
    vector<int> b(K * N);
    vector<int> res(M * N ,0);

    for(int i = 0; i < M; i++)
        for(int j = 0; j < K; j++)
            a[i * K + j] = i + 1;

    for(int i = 0; i < K; i++)
        for(int j = 0; j < N; j++)
            b[i * N + j] = j + 1;   

    cl::Buffer buff1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * M * K, a.data());

    cl::Buffer buff2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * K * N, b.data());

    cl::Buffer resbuff(context, CL_MEM_READ_WRITE,
                       sizeof(int) * M * N);

    cl::Program program(context, util::loadProgram("matmul_local.cl"));
    program.build({device});
    
    cl::Kernel kernel(program,"matmul_local");
    kernel.setArg(0, buff1);
    kernel.setArg(1, buff2);
    kernel.setArg(2, resbuff);
    kernel.setArg(3, M);
    kernel.setArg(4, K);
    kernel.setArg(5, N);

    // TILE=2 → local = (2,2)
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange,
                               cl::NDRange(M, N),
                               cl::NDRange(2, 2));  

    queue.enqueueReadBuffer(resbuff, CL_TRUE, 0,
                            sizeof(int) * M * N, res.data());

    cout << endl << "A matrix:" << endl;
    for(int i = 0 ; i < M ; i++)
    {
        for(int j = 0 ; j < K ; j++)
            cout << a[i * K + j] << " ";
        cout << endl;
    }

    cout << endl << "B matrix:" << endl;
    for(int i = 0 ; i < K ; i++)
    {
        for(int j = 0 ; j < N ; j++)
            cout << b[i * N + j] << " ";
        cout << endl;
    }

    cout << endl << "Result matrix :" << endl;
    for(int i = 0 ; i < M ; i++)
    {
        for(int j = 0 ; j < N ; j++)
            cout << res[i * N + j] << " ";
        cout << endl;
    }
}
