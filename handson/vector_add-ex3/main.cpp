#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define WORK_ITEMS 24

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

    vector<int> a(WORK_ITEMS);
    vector<int> b(WORK_ITEMS);
    vector<int> c(WORK_ITEMS,0);

    for(int i = 0 ; i < WORK_ITEMS; i++)
    {
        a[i] = i + 1;
        b[i] = i + 2;
    }

    cl::Buffer a_buffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int)* WORK_ITEMS, a.data());
    cl::Buffer b_buffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS,b.data());
    cl::Buffer c_buffer(context,CL_MEM_READ_WRITE,sizeof(int) * WORK_ITEMS);
    cl::Program program(context,util::loadProgram("vector_add.cl"));
    program.build({device});

    cl::Kernel kernel(program,"vector_add");
    kernel.setArg(0,a_buffer);
    kernel.setArg(1,b_buffer);
    kernel.setArg(2,c_buffer);

    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(WORK_ITEMS));
    queue.enqueueReadBuffer(c_buffer,CL_TRUE,0,sizeof(int) * WORK_ITEMS,c.data());

    for(int i = 0 ; i < WORK_ITEMS; i++)
    {
        cout << "Add result [" << i << "]: " <<c[i] <<endl;
    }
}