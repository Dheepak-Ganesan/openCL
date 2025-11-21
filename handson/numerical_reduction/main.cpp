#define __ENABLE_CL_EXTENSION
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#include "cl.hpp"
#include "util.hpp"

#define WORK_ITEMS 512
#define WORK_GROUP 256

int main()
{
    // Get platform
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    // Get device
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    // Context + Profiling Queue
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    vector<float> host_buffer1(WORK_ITEMS);
    vector<float> host_buffer2(WORK_ITEMS);
    vector<float> host_res(1,0);

    for(int i = 0; i<WORK_ITEMS;i++)
    {
        host_buffer1[i] = i + 1.0;
        host_buffer2[i] = i + 5.0;
    }

    cl::Buffer buffer1_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(float)*host_buffer1.size(),host_buffer1.data());
    cl::Buffer buffer2_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(float)* host_buffer2.size(),host_buffer2.data());
    cl::Buffer res_dev(context,CL_MEM_WRITE_ONLY,sizeof(float) * 1);

    cl::Program program(context,util::loadProgram("local_mem.cl"));
    program.build({device});
    cl::Kernel kernel(program,"compute_local");
    cl::Event event;
    kernel.setArg(0,buffer1_dev);
    kernel.setArg(1,buffer2_dev);
    kernel.setArg(2,res_dev);
    kernel.setArg(3,cl::Local(sizeof(float) * WORK_GROUP));

    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(WORK_ITEMS),cl::NDRange(WORK_GROUP),nullptr,&event);
    event.wait();
    cl_ulong st = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double timetaken = (end-st) * 1e-6;

    queue.enqueueReadBuffer(res_dev,CL_TRUE,0,sizeof(float) * 1,host_res.data());

    for(int i = 0;i < 1; i++)
    {
        cout << "result [" << i << "] " << host_res[i] <<endl;
    }

    cout<< "total time taken  : " << timetaken << endl;

}

  //  1 2 3  4 5 6 7 8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 = work group size 
    //64 work item size
   // stride = 16
    //stride/=2 8,4,2,1
    //1+17,2+18,..