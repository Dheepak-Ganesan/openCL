#define __ENABLE_CL_EXTENSION
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#include "cl.hpp"
#include "util.hpp"

#define WORK_ITEMS 1024
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

    // Allocate host memory
    vector<int> a_host(WORK_ITEMS);
    vector<int> b_host(WORK_ITEMS);
    vector<int> res_host_global(WORK_ITEMS, 0);
    vector<int> res_host_local(WORK_ITEMS, 0);

    //fill data
    for(int i = 0 ; i < WORK_ITEMS ; i++)
    {
        a_host[i] = i;
        b_host[i] = i + 1;
    }

    // Allocate device memory
    cl::Buffer a_device_global(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(int) * WORK_ITEMS, a_host.data());

    cl::Buffer b_device_global(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(int) * WORK_ITEMS, b_host.data());

    cl::Buffer res_device_global(context, CL_MEM_WRITE_ONLY,
                                 sizeof(int) * WORK_ITEMS);

    cl::Buffer a_device_local(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(int) * WORK_ITEMS, a_host.data());

    cl::Buffer b_device_local(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(int) * WORK_ITEMS, b_host.data());

    cl::Buffer res_device_local(context, CL_MEM_WRITE_ONLY,
                                sizeof(int) * WORK_ITEMS);

    // Load programs
    cl::Program program_global(context, util::loadProgram("compute_global.cl"));
    cl::Program program_local(context, util::loadProgram("compute_local.cl"));

    program_global.build({device});
    program_local.build({device});

    // Create kernels
    cl::Kernel kernel_global(program_global, "compute_global");
    cl::Kernel kernel_local(program_local, "compute_local");

    // Set arguments for global kernel
    kernel_global.setArg(0, a_device_global);
    kernel_global.setArg(1, b_device_global);
    kernel_global.setArg(2, res_device_global);

    // Set arguments for local kernel
    kernel_local.setArg(0, a_device_local);
    kernel_local.setArg(1, b_device_local);
    kernel_local.setArg(2, res_device_local);
    kernel_local.setArg(3, cl::Local(sizeof(int) * WORK_GROUP));
    kernel_local.setArg(4, cl::Local(sizeof(int) * WORK_GROUP));

    // Events
    cl::Event event_global, event_local;

    /******** GLOBAL KERNEL ********/
    queue.enqueueNDRangeKernel(kernel_global,
                               cl::NullRange,
                               cl::NDRange(WORK_ITEMS),
                               cl::NDRange(WORK_GROUP),
                               nullptr,
                               &event_global);
    queue.enqueueReadBuffer(res_device_global, CL_TRUE, 0,
                            sizeof(int) * WORK_ITEMS, res_host_global.data());

    /******** LOCAL KERNEL ********/
    queue.enqueueNDRangeKernel(kernel_local,
                               cl::NullRange,
                               cl::NDRange(WORK_ITEMS),
                               cl::NDRange(WORK_GROUP),
                               nullptr,
                               &event_local);
    queue.enqueueReadBuffer(res_device_local, CL_TRUE, 0,
                            sizeof(int) * WORK_ITEMS, res_host_local.data());

    /******** PROFILE BOTH ********/
    event_global.wait();
    cl_ulong g_start = event_global.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong g_end   = event_global.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double g_time_ms = (g_end - g_start) * 1e-6;

    event_local.wait();
    cl_ulong l_start = event_local.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong l_end   = event_local.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double l_time_ms = (l_end - l_start) * 1e-6;

    cout << "Global Kernel Time = " << g_time_ms << " ms\n";
    cout << "Local  Kernel Time = " << l_time_ms << " ms\n";

    // Print sample output
    cout << "\nGLOBAL Result (first 20):\n";
    for(int i = 0; i < 10; i++)
        cout << "res_global[" << i << "] = " << res_host_global[i] << endl;

    cout << "\nLOCAL Result (first 20):\n";
    for(int i = 0; i < 20; i++)
        cout << "res_local[" << i << "] = " << res_host_local[i] << endl;

    return 0;
}
