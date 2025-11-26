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
    vector<int> d(WORK_ITEMS,0);
    vector<int> e(WORK_ITEMS);
    vector<int> f(WORK_ITEMS,0);
    vector<int> g(WORK_ITEMS);

    for(int i = 0 ; i < WORK_ITEMS; i++)
    {
        a[i] = i + 1;
        b[i] = i + 1;
        e[i] = i + 1;
        g[i] = i + 1;
    }

    cl::Buffer a_buffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int)* WORK_ITEMS, a.data());
    cl::Buffer b_buffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS,b.data());
    cl::Buffer c_buffer(context,CL_MEM_READ_WRITE,sizeof(int) * WORK_ITEMS);
    cl::Buffer d_buffer(context,CL_MEM_READ_WRITE, sizeof(int)*WORK_ITEMS);
    cl::Buffer e_buffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS,e.data());
    cl::Buffer f_buffer(context,CL_MEM_READ_WRITE, sizeof(int)*WORK_ITEMS);
    cl::Buffer g_buffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int)* WORK_ITEMS, g.data());

    cl::Program program(context,util::loadProgram("chaining_vectors.cl"));
    program.build({device});

    cl::Kernel kernel_c(program,"chain_add");
    kernel_c.setArg(0,a_buffer);
    kernel_c.setArg(1,b_buffer);
    kernel_c.setArg(2,c_buffer);

    queue.enqueueNDRangeKernel(kernel_c,cl::NullRange,cl::NDRange(WORK_ITEMS));
    queue.enqueueReadBuffer(c_buffer,CL_TRUE,0,sizeof(int) * WORK_ITEMS,c.data());
    

    cl::Kernel kernel_d(program,"chain_add");
    kernel_d.setArg(0,c_buffer);
    kernel_d.setArg(1,e_buffer);
    kernel_d.setArg(2,d_buffer);
    queue.enqueueNDRangeKernel(kernel_d,cl::NullRange,cl::NDRange(WORK_ITEMS));
    queue.enqueueReadBuffer(d_buffer,CL_TRUE,0,sizeof(int) * WORK_ITEMS,d.data());

    cl::Kernel kernel_f(program,"chain_add");
    kernel_f.setArg(0,d_buffer);
    kernel_f.setArg(1,g_buffer);
    kernel_f.setArg(2,f_buffer);
    queue.enqueueNDRangeKernel(kernel_f,cl::NullRange,cl::NDRange(WORK_ITEMS));
    queue.enqueueReadBuffer(f_buffer,CL_TRUE,0,sizeof(int) * WORK_ITEMS,f.data());


    for(int i = 0 ; i < WORK_ITEMS; i++)
    {
        cout << "C result [" << i << "]: " <<c[i] <<endl;
    }

    for(int i = 0 ; i < WORK_ITEMS; i++)
    {
        cout << "D result [" << i << "]: " <<d[i] <<endl;
    }

    for(int i = 0 ; i < WORK_ITEMS; i++)
    {
        cout << "F result [" << i << "]: " <<f[i] <<endl;
    }
}