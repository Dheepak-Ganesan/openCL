#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <cstdlib> 
using namespace std;

#define WORK_ITEMS 512
#define WORK_GROUP 64
#define WG_COUNT (WORK_ITEMS / WORK_GROUP)

int main() {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    vector<int> host_input(WORK_ITEMS);
    vector<int> host_min_res(WG_COUNT);
    vector<int> host_max_res(WG_COUNT);

    for(int i = 0; i < WORK_ITEMS; i++)
        host_input[i] = rand() % 1000;

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * WORK_ITEMS, host_input.data());
    cl::Buffer buffer_min(context, CL_MEM_WRITE_ONLY, sizeof(int) * WG_COUNT);
    cl::Buffer buffer_max(context, CL_MEM_WRITE_ONLY, sizeof(int) * WG_COUNT);

    cl::Program program(context, util::loadProgram("min_max_1d.cl"));
    program.build({device});

    // Kernel object for min
    cl::Kernel kernel_min(program, "min_compute");
    kernel_min.setArg(0, buffer_input);
    kernel_min.setArg(1, buffer_min);
    kernel_min.setArg(2, cl::Local(sizeof(int) * WORK_GROUP));

    queue.enqueueNDRangeKernel(kernel_min, cl::NullRange,
                               cl::NDRange(WORK_ITEMS),
                               cl::NDRange(WORK_GROUP));
    queue.finish();

    queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0,
                            sizeof(int) * WG_COUNT, host_min_res.data());

    // Kernel object for max
    cl::Kernel kernel_max(program, "max_compute");
    kernel_max.setArg(0, buffer_input);
    kernel_max.setArg(1, buffer_max);
    kernel_max.setArg(2, cl::Local(sizeof(int) * WORK_GROUP));

    queue.enqueueNDRangeKernel(kernel_max, cl::NullRange,
                               cl::NDRange(WORK_ITEMS),
                               cl::NDRange(WORK_GROUP));
    queue.finish();

    queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0,
                            sizeof(int) * WG_COUNT, host_max_res.data());

    cout << "Per-workgroup minimums:" << endl;
    for(int i = 0; i < WG_COUNT; i++)
        cout << "min_res[" << i << "] = " << host_min_res[i] << endl;

    int global_min = host_min_res[0];
    for(int i = 1; i < WG_COUNT; i++)
        if(host_min_res[i] < global_min) global_min = host_min_res[i];
    cout << "Global Min: " << global_min << endl;

    cout << "Per-workgroup maximums:" << endl;
    for(int i = 0; i < WG_COUNT; i++)
        cout << "max_res[" << i << "] = " << host_max_res[i] << endl;

    int global_max = host_max_res[0];
    for(int i = 1; i < WG_COUNT; i++)
        if(host_max_res[i] > global_max) global_max = host_max_res[i];
    cout << "Global Max: " << global_max << endl;

    return 0;
}
