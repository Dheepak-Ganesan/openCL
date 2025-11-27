#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main()
{
    const int HEIGHT = 6;
    const int WIDTH  = 6;

    const int KH = 3;
    const int KW = 3;

    const int outH = HEIGHT - KH + 1;
    const int outW = WIDTH  - KW + 1;

    const int WG = 3;  // work-group (tile) size
    const int patch = WG + KH - 1;

    vector<int> input(HEIGHT * WIDTH);
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
            input[i * WIDTH + j] = i + 1;

    vector<int> kernel_host(KH * KW, 1);
    vector<int> output(outH * outW, 0);

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context({device});
    cl::CommandQueue queue(context, device);

    cl::Buffer input_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(int) * input.size(), input.data());

    cl::Buffer kernel_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(int) * kernel_host.size(), kernel_host.data());

    cl::Buffer out_dev(context, CL_MEM_WRITE_ONLY,
                       sizeof(int) * output.size());

    cl::Program program(context, util::loadProgram("conv2d_local.cl"));
    program.build({device});

    cl::Kernel ker(program, "conv2d_local");

    ker.setArg(0, input_dev);
    ker.setArg(1, kernel_dev);
    ker.setArg(2, out_dev);
    ker.setArg(3, KH);
    ker.setArg(4, KW);
    ker.setArg(5, outH);
    ker.setArg(6, outW);
    ker.setArg(7, HEIGHT);
    ker.setArg(8, WIDTH);
    ker.setArg(9, WG);
    ker.setArg(10, patch);

    cl::NDRange global(HEIGHT, WIDTH);
    cl::NDRange local(WG, WG);

    queue.enqueueNDRangeKernel(ker, cl::NullRange, global, local);
    queue.enqueueReadBuffer(out_dev, CL_TRUE, 0,
                            sizeof(int) * output.size(), output.data());

    cout << "Output:\n";
    for (int i = 0; i < outH; i++)
    {
        for (int j = 0; j < outW; j++)
            cout << output[i * outW + j] << " ";
        cout << "\n";
    }

    return 0;
}
