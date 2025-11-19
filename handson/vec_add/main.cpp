// step 1 : headers(cpp &openCL)
#include "cl.hpp"
#include "util.hpp"
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<string>
#include<vector>
#include<fstream>
using namespace std;

#define _ENABLE_CL_EXCEPTIONS

// step 2: work items(global dimensions)
#define GLOBAL_SIZE 256

int main()
{
    //step 3: set the platform
    std::vector<cl::Platform> num_platforms;
    cl::Platform::get(&num_platforms);

    std::cout<< " number of platforms available: " << num_platforms.size() <<endl;

    cl::Platform platform = num_platforms[0];

    //step 4: set the device(choose which device to work with)
    std::vector<cl::Device> num_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&num_devices);
    cl::Device device = num_devices[0];
    std::cout << " number of devices available : " << num_devices.size() <<endl;

    //step 5: create context(its a coepnCL container where we can combine devices)
    cl::Context context(device);

    //step 6: To pass the commands through queue(copying memory, running kernels,etc.. everything goes through a command queue)
    cl::CommandQueue CommandQueue(context,device);

    //step 7: Allocate device memory buffers

    //host arrays
    std::vector<int> host_a(GLOBAL_SIZE);
    std::vector<int> host_b(GLOBAL_SIZE);
    std::vector<int> host_result(GLOBAL_SIZE,0);

    //input data
    for(int i = 0 ;i<GLOBAL_SIZE;i++)
    {
        host_a[i] = i + 1;
        host_b[i] = i + 2;
    }

    //Device buffer
    cl::Buffer device_a(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * GLOBAL_SIZE,host_a.data());
    cl::Buffer device_b(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * GLOBAL_SIZE, host_b.data());
    cl::Buffer device_result(context, CL_MEM_WRITE_ONLY,sizeof(int) * GLOBAL_SIZE);


    //step 8: write kernel code - add.cl(DONE!)

    //step 9: Load the kernel source
    std::string kernel_source = util::loadProgram("add.cl");
    cout<<"----------------------------------------------------------------------"<<endl;
    cout<<"Kernel code which will compile in the selected device on run time:"<<endl;
    cout << kernel_source<<endl;
    cout<<"----------------------------------------------------------------------"<<endl;

    //step 10: create a program object and build the program in specified device
    cl::Program program_obj(context,kernel_source); // (context, code in string)
    program_obj.build({device});

    //step 11: similar to program object(host), create a kernel object(device)
    cl::Kernel kernel_add(program_obj,"eltwise_add"); // (program object, function name)
    
    //step 12: set arguments to the kernel
    kernel_add.setArg(0,device_a);
    kernel_add.setArg(1,device_b);
    kernel_add.setArg(2,device_result);
    kernel_add.setArg(3,GLOBAL_SIZE);

    //step 13: enqueue kernel (execute the kernel in the device)
    cl::NDRange global(GLOBAL_SIZE);
    CommandQueue.enqueueNDRangeKernel(kernel_add,cl::NullRange,global,cl::NullRange);
    CommandQueue.finish(); //blocks CPU until the work is done

    //step 14: Read back to host device 
    CommandQueue.enqueueReadBuffer(device_result,CL_TRUE,0,sizeof(int) * GLOBAL_SIZE,host_result.data());

    //step 15: print and verify
    for(int i = 0;i<GLOBAL_SIZE;i++)
    {
        cout << "host_result[" << i << "] :" << host_result[i] << endl;
    }   

}


// Recap of Full Flow:

// Discover platforms & devices
// Create context
// Create command queue
// Allocate host & device buffers
// Load kernel source
// Build program object
// Create kernel object
// Set kernel arguments
// Enqueue kernel execution
// Read back results
// Verify results