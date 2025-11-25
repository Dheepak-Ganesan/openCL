#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define HEIGHT 64
#define WIDTH  64
#define DEPTH  4

#define WORK_ITEMS (HEIGHT * WIDTH * DEPTH)

#define LOCAL1 (HEIGHT / 8)
#define LOCAL2 (WIDTH  / 8)
#define LOCAL3 (DEPTH  / 2)

#define WG_COUNT ((HEIGHT / LOCAL1) * (WIDTH / LOCAL2) * (DEPTH / LOCAL3))
#define WG_SIZE  (LOCAL1 * LOCAL2 * LOCAL3)

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
    vector<int> host_max_res(WG_COUNT);

    for(int i = 0; i < WORK_ITEMS; i++)
        host_input[i] = (rand() % 100);

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * WORK_ITEMS, host_input.data());

    cl::Buffer buffer_max(context, CL_MEM_WRITE_ONLY,
                          sizeof(int) * WG_COUNT);

    cl::Program program(context, util::loadProgram("max_3d.cl"));
    program.build({device});

    cl::Kernel kernel_max(program, "max_compute");
    kernel_max.setArg(0, buffer_input);
    kernel_max.setArg(1, buffer_max);
    kernel_max.setArg(2, cl::Local(sizeof(int) * WG_SIZE));
    kernel_max.setArg(3, HEIGHT);
    kernel_max.setArg(4, WIDTH);
    kernel_max.setArg(5, DEPTH);

    queue.enqueueNDRangeKernel(
        kernel_max,
        cl::NullRange,
        cl::NDRange(HEIGHT, WIDTH, DEPTH),
        cl::NDRange(LOCAL1, LOCAL2, LOCAL3)
    );

    queue.finish();

    queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0,
                            sizeof(int) * WG_COUNT,
                            host_max_res.data());

    cout << "Per-workgroup maximums:" << endl;
    for(int i = 0; i < WG_COUNT; i++)
        cout << "max_res[" << i << "] = " << host_max_res[i] << endl;

    int global_max = host_max_res[0];
    for(int i = 1; i < WG_COUNT; i++)
        if(host_max_res[i] > global_max)
            global_max = host_max_res[i];

    cout << "Global Max: " << global_max << endl;

    return 0;
}
