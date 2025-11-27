#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
using namespace std;

#define HEIGHT 4
#define WIDTH 4
#define kH 2
#define kW 2
#define outH (HEIGHT - kH + 1)
#define outW (WIDTH - kW + 1)

int main() {

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    vector<int> input(HEIGHT * WIDTH);
    for(int i = 0; i< HEIGHT; i++)
    {
        for(int j = 0 ; j<WIDTH; j++)
        {
            input[i * WIDTH + j] = i+1;
        }
    }

    vector<int> k(kH * kW,1); 

    vector<int> output(outH * outW,0);

    cl::Buffer input_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * input.size(),input.data());
    cl::Buffer kernel_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * k.size(),k.data());
    cl::Buffer out_dev(context, CL_MEM_WRITE_ONLY,sizeof(int) * output.size());

    cl::Program program(context, util::loadProgram("conv2d.cl"));
    program.build({device});

    cl::Kernel kernel_conv2d(program, "conv2d");
    kernel_conv2d.setArg(0,input_dev);
    kernel_conv2d.setArg(1, kernel_dev);
    kernel_conv2d.setArg(2,out_dev);
    kernel_conv2d.setArg(3, kH);
    kernel_conv2d.setArg(4, kW);
    kernel_conv2d.setArg(5,outH);
    kernel_conv2d.setArg(6,outW);
    kernel_conv2d.setArg(7,HEIGHT);
    kernel_conv2d.setArg(8,WIDTH);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel_conv2d,cl::NullRange,cl::NDRange(outH,outW),cl::NullRange,nullptr,&event);
    event.wait();

    queue.enqueueReadBuffer(out_dev, CL_TRUE, 0, sizeof(int) * output.size(),output.data());
    for(int i = 0 ; i < outH;i++)
    {
        for(int j = 0 ; j < outW;j++)
        {
            cout <<output[i * outW + j] << " ";
        }
        cout << endl;
    }
    
}
