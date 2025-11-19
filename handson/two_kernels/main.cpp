    #include "cl.hpp"
    #include "util.hpp"
    #define _ENABLE_CL_EXCEPTIONS

    #include <iostream>
    #include <cstdio>
    #include <cstdlib>
    #include <string>
    #include <vector>
    using namespace std;

    #define WORK_ITEMS 512

    int main()
    {
        std::vector<cl::Platform> platforms_list;
        cl::Platform::get(&platforms_list);

        cl::Platform platform_sub = platforms_list[0];


        std::vector<cl::Device> devices_list;
        platform_sub.getDevices(CL_DEVICE_TYPE_ALL,&devices_list);

        cl::Device device_sub = devices_list[0];

        cl::Context context_sub(device_sub);
        cl::CommandQueue queue_sub(context_sub,device_sub);

        std::vector<int> host_first(WORK_ITEMS,50);
        std::vector<int> host_second(WORK_ITEMS,25);
        std::vector<int> host_third(WORK_ITEMS,15);
        std::vector<int> host_fourth(WORK_ITEMS,10);
        std::vector<int> host_result_array(WORK_ITEMS,0);

        cl::Buffer device_first(context_sub,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS, host_first.data());
        cl::Buffer device_second(context_sub,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS, host_second.data());
        cl::Buffer device_third(context_sub,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS, host_third.data());
        cl::Buffer device_fourth(context_sub,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * WORK_ITEMS,host_fourth.data());
        cl::Buffer device_result_array(context_sub,CL_MEM_WRITE_ONLY,sizeof(int) * WORK_ITEMS);

        cl::Program program_sub(context_sub,util::loadProgram("sub.cl"));
        cl::Program program_add(context_sub,util::loadProgram("add.cl"));

        program_sub.build({device_sub});
        program_add.build({device_sub});

        cl::Kernel kernel_sub(program_sub,"eltwise_sub");
        kernel_sub.setArg(0,device_first);
        kernel_sub.setArg(1,device_second);
        kernel_sub.setArg(2,device_third);
        kernel_sub.setArg(3,device_fourth);
        kernel_sub.setArg(4,device_result_array);
        kernel_sub.setArg(5,256);

        cl::Kernel kernel_add(program_add,"eltwise_add");
        kernel_add.setArg(0,device_first);
        kernel_add.setArg(1,device_second);
        kernel_add.setArg(2,device_result_array);
        kernel_add.setArg(3,256);
        kernel_add.setArg(4,256);

        cl::NDRange global(WORK_ITEMS/2);
        queue_sub.enqueueNDRangeKernel(kernel_sub,cl::NullRange,global, cl::NullRange );
        
        // queue_sub.enqueueReadBuffer(device_result_array,CL_TRUE,0,sizeof(int)* (WORK_ITEMS/2), host_result_array.data());

        queue_sub.enqueueNDRangeKernel(kernel_add,cl::NullRange,global, cl::NullRange );
        
        queue_sub.enqueueReadBuffer(device_result_array,CL_TRUE,0,sizeof(int)* (WORK_ITEMS), host_result_array.data());

        for(int i = 0; i < WORK_ITEMS  ; i++)
            cout<<" Result array[" << i << "] " << host_result_array[i] <<endl;

    }

// Description:
// This program demonstrates how to run two different OpenCL kernels (subtraction and addition) on the same device buffer and store the results in different halves of the output array.
// Creates OpenCL context & command queue for the first available device (CPU/GPU/NPU).
// Initializes host arrays (input data and result buffer).
// Creates OpenCL buffers for inputs and outputs.
// Builds two separate OpenCL programs (sub.cl and add.cl).
// Sets kernel arguments for:
// eltwise_sub → writes results to indices 0–255
// eltwise_add → writes results to indices 256–511
// Enqueues kernels sequentially, each processing 256 work-items.
// Reads back the full result buffer.
// Prints all 512 output values.
// So effectively:
// Kernel 1 computes:
// res[i] = (a[i] - b[i]) - (c[i] - d[i]) for i = 0..255

// Kernel 2 computes:
// res[i+256] = a[i] + b[i] for i = 0..255

// Both results are stored in the same output buffer, in different regions.