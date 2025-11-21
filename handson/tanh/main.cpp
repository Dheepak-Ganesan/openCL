#define __ENABLE_CL_EXCEPTIONS
#include <iostream>
#include <vector>
#include "cl.hpp"
#include "util.hpp"
using namespace std;
#include <cstdlib>

#define WORK_ITEMS 256

int main()
{
    vector<cl::Platform> platform_list;
    cl::Platform::get(&platform_list);

    cl::Platform platform = platform_list[0];

    vector<cl::Device> devices_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&devices_list);

    cl::Device device = devices_list[0];

    cl::Context context(device);
    cl::CommandQueue queue(context,device);

    vector<float> one_host(WORK_ITEMS);
    for(int i = 0 ; i <WORK_ITEMS; i++)
    {
        one_host[i] = ((rand() % 101) - 50) / 10.0f;  // -5.0 to 5.0
    }

    cl::Buffer buff_dev(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * WORK_ITEMS,one_host.data());

    cl::Program program(context,util::loadProgram("tanh.cl"));
    program.build({device});

    cl::Kernel kernel(program,"eltwise_tanh");
    kernel.setArg(0,buff_dev);

    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(WORK_ITEMS),cl::NullRange);
    queue.enqueueReadBuffer(buff_dev,CL_TRUE,0,sizeof(float) * WORK_ITEMS,one_host.data());

    cout << "Tanh results " << endl;
    for(int i = 0 ; i< WORK_ITEMS ; i++)
    {
        cout << "Result[" << i <<"] " << one_host[i] << endl;
    }
}