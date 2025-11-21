#define __ENABLE_CL_EXCEPTIONS
#include <iostream>
#include <vector>
#include "cl.hpp"
#include "util.hpp"
using namespace std;

#define M 3
#define K 4
#define N 3
int main()
{

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device,CL_QUEUE_PROFILING_ENABLE);

    vector<int> matA(M * K);
    vector<int> matB(K * N);
    vector<int> matC(M * N,0);

    for(int i = 0; i<M; i ++)
    {
        for(int j = 0; j<K; j++)
        {
            matA[i *K + j] = i + 1;
        }
    }

    for(int i = 0; i<K; i ++)
    {
        for(int j = 0; j<N; j++)
        {
            matB[i *N + j] = i + 2;
        }
    }

    cl::Buffer buff1(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (M * K ),matA.data());
    cl::Buffer buff2(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (K*N),matB.data());
    cl::Buffer buff3(context,CL_MEM_WRITE_ONLY, sizeof(int) * (M*N));

    cl::Program program(context, util::loadProgram("matmul.cl"));
    program.build({device});

    cl::Kernel kernel(program,"matmul");
    kernel.setArg(0,buff1);
    kernel.setArg(1,buff2);
    kernel.setArg(2,buff3);
    kernel.setArg(3, M);
    kernel.setArg(4, K);
    kernel.setArg(5, N);


    cl::Event event;
    queue.enqueueNDRangeKernel(kernel,cl::NullRange, cl::NDRange(M,N),cl::NullRange,nullptr,&event);
    event.wait();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double timetaken = (end - start) * 1e-6;

    queue.enqueueReadBuffer(buff3,CL_TRUE,0,sizeof(int) * (M*N),matC.data());

    for(int i = 0;i<M;i++)
    {
        for(int j = 0 ; j<N; j++)
        {
            cout<< matC[i * N + j] << " " ;
        }
        cout << endl;
    }

}
