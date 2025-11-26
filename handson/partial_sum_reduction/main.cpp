#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
using namespace std;

int main() {

    int num_steps = 40;

    // Select platform/device
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    const int global_size = 20;  // multiple of local size
    const int local_size = 4;

    // iterations for wi = 40/20 = 2

    int num_groups = global_size / local_size;
    vector<int> partial_sums(num_groups);

    vector<int> input(num_steps,1);
    cl::Buffer input_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * num_steps,input.data());
    cl::Buffer partial_buf(context, CL_MEM_WRITE_ONLY,sizeof(int) * num_groups);

    // build program
    cl::Program program(context, util::loadProgram("partial_reduction.cl"));
    program.build({device});

    cl::Kernel kernel(program, "reduction");
    kernel.setArg(0,input_dev);
    kernel.setArg(1, partial_buf);
    kernel.setArg(2,cl::Local(sizeof(int) *local_size));
    kernel.setArg(3, num_steps);

    cl::Event event;
    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(global_size),
        cl::NDRange(local_size),
        nullptr,
        &event
    );
    event.wait();

    queue.enqueueReadBuffer(partial_buf, CL_TRUE, 0,
                            sizeof(int) * num_groups,
                            partial_sums.data());

       int host_sum = 0;                     
      for(int i = 0; i < num_groups;i++)
      {
            host_sum+=partial_sums[i]; // can also be summed up by stride halving
      }                      

    cout << "Total sum: " << host_sum <<endl;
    
}
