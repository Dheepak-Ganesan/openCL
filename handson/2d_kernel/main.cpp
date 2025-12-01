#define __ENABLE_CL_EXTENSION
#include <iostream>
#include <vector>
#include "cl.hpp"
#include "util.hpp"

using namespace std;

#define WIDTH  4   // columns
#define HEIGHT 4   // rows

int main() {
    // --- Platform and device ---
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    vector<int> mat1_host(WIDTH * HEIGHT,20);
    vector<int> mat2_host(WIDTH * HEIGHT,10);
    vector<int> result_host(WIDTH * HEIGHT);
    cl::Buffer buff1_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WIDTH * HEIGHT,mat1_host.data());
    cl::Buffer buff2_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WIDTH * HEIGHT,mat2_host.data());
    cl::Buffer resbuff_dev(context,CL_MEM_READ_WRITE,sizeof(int) * WIDTH * HEIGHT);

    cl::Program program(context,util::loadProgram("matrix_add.cl"));
    program.build({device});

    cl::Kernel kernel(program,"matrix_add");
    kernel.setArg(0,buff1_dev);
    kernel.setArg(1,buff2_dev);
    kernel.setArg(2,resbuff_dev);
    kernel.setArg(3,WIDTH);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(WIDTH,HEIGHT),cl::NullRange,nullptr,&event);
    event.wait();
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double timetaken = (end - start) * 1e-6;
    queue.enqueueReadBuffer(resbuff_dev,CL_TRUE,0,sizeof(int) * HEIGHT * WIDTH,result_host.data());

    for(int i = 0 ; i < HEIGHT;i++)
    {
        for(int j = 0;j < WIDTH; j++)
        {
            int idx = i * WIDTH + j;
            cout << "result [" << i <<"]" <<"[" << j <<"]: " << result_host[idx] << endl;
        }
    }

    cout << " Total time taken : " << timetaken << endl;

}