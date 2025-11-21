#define __ENABLE_CL_EXCEPTIONS
#include <iostream>
#include <vector>
#include <cstdlib>
#include "cl.hpp"
#include "util.hpp"
using namespace std;

#define WORK_ITEMS 512
int main()
{
    vector<cl::Platform> platforms_list;
    cl::Platform::get(&platforms_list);
    cl::Platform platform = platforms_list[0];

    vector<cl::Device> devices_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&devices_list);
    cl::Device device = devices_list[0];

    cl::Context context(device);
    cl::CommandQueue queue(context,device,CL_QUEUE_PROFILING_ENABLE);

    vector<float> host_array(WORK_ITEMS);
    for(int i = 0;i< WORK_ITEMS;i++)
    {
        host_array[i] = ( rand() %20);
    }

    cl::Buffer device_buffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,sizeof(float)*WORK_ITEMS,host_array.data());
    cl::Program program(context,util::loadProgram("relu_6.cl"));
    program.build({device});
   
    cl::Kernel kernel(program,"eltwise_relu6");
    kernel.setArg(0,device_buffer);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(WORK_ITEMS),cl::NullRange,nullptr,&event);
    event.wait();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double timetaken = (end - start) * 1e-6;

    queue.enqueueReadBuffer(device_buffer,CL_TRUE,0,sizeof(float)*WORK_ITEMS,host_array.data());

    cout<< "RELU6 Results" << endl;
    for(int i = 0 ; i<WORK_ITEMS; i++)
    {
        cout << "Result [" << i <<"] " << host_array[i] << endl;
    }

    cout << "Total time taken for RELU6 : " << timetaken << endl;
}