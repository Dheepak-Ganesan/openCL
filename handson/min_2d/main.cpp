#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define HEIGHT 64
#define WIDTH  64
#define WORK_ITEMS (HEIGHT * WIDTH)

#define LOCAL1 (HEIGHT / 8)
#define LOCAL2 (WIDTH  / 8)
#define WG_COUNT ((HEIGHT / LOCAL1) * (WIDTH / LOCAL2))

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

    for(int i = 0; i < WORK_ITEMS; i++)
        host_input[i] = (rand() % 100) - 10;

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(int) * WORK_ITEMS, host_input.data());

    cl::Buffer buffer_min(context, CL_MEM_WRITE_ONLY,
                          sizeof(int) * WG_COUNT);

    cl::Program program(context, util::loadProgram("min_2d.cl"));
    program.build({device});

    int local_size = LOCAL1 * LOCAL2;

    cl::Kernel kernel_min(program, "min_compute");
    kernel_min.setArg(0, buffer_input);
    kernel_min.setArg(1, buffer_min);
    kernel_min.setArg(2, cl::Local(sizeof(int) * local_size));
    kernel_min.setArg(3, WIDTH);

    queue.enqueueNDRangeKernel(kernel_min,
                               cl::NullRange,
                               cl::NDRange(HEIGHT, WIDTH),
                               cl::NDRange(LOCAL1, LOCAL2));

    queue.finish();

    queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0,
                            sizeof(int) * WG_COUNT,
                            host_min_res.data());

    cout << "Per-workgroup minimums:" << endl;
    for(int i = 0; i < WG_COUNT; i++)
        cout << "min_res[" << i << "] = " << host_min_res[i] << endl;

    int global_min = host_min_res[0];
    for(int i = 1; i < WG_COUNT; i++)
        if(host_min_res[i] < global_min)
            global_min = host_min_res[i];

    cout << "Global Min: " << global_min << endl;

    return 0;
}
