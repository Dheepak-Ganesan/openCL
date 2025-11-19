#include "cl.hpp"
#include "util.hpp"
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<string>
#include<vector>
using namespace std;
#define _ENABLE_CL_EXCEPTIONS

//flow
// global mem -> platforms/devices -> context -> command queue -> buffer(global and host) ->kernel code -> 
// program buffer -> build -> kernel object ->set kernel argument ->enqueueNDRangeKernel -> read from device -> test

#define GLOBAL_SIZE 1024 //no.of work items

int main()
{
    std::vector<cl::Platform> platform_array;
    cl::Platform::get(&platform_array);

    std::cout<< "number of platforms available : "<< platform_array.size() <<endl;

    cl::Platform platform = platform_array[0];

    std::vector<cl::Device> device_array;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&device_array);

    cl::Device device = device_array[0];

    cl::Context context(device);
    cl::CommandQueue commandqueue_mult(context,device);

    //host buffers
    std::vector<float> host_one(GLOBAL_SIZE,1);
    std::vector<float> host_two(GLOBAL_SIZE,-1);
    std::vector<float> host_result(GLOBAL_SIZE,0);

    //device buffers

    cl::Buffer device_one(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * GLOBAL_SIZE, host_one.data());
    cl::Buffer device_two(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * GLOBAL_SIZE, host_two.data());

    cl::Buffer device_result(context,CL_MEM_WRITE_ONLY, sizeof(float) * GLOBAL_SIZE);

    // std::string kernel_source =  util::loadProgram("vecmul.cl");
    cl::Program program_obj(context,util::loadProgram("vecmul.cl"));
    program_obj.build({device});

    cl::Kernel kernel_obj(program_obj,"eltwise_mul");
    kernel_obj.setArg(0,device_one);
    kernel_obj.setArg(1,device_two);
    kernel_obj.setArg(2,device_result);
    kernel_obj.setArg(3,GLOBAL_SIZE);

    cl::NDRange global = GLOBAL_SIZE;
    commandqueue_mult.enqueueNDRangeKernel(kernel_obj,cl::NullRange,global,cl::NullRange);
    commandqueue_mult.finish();
    commandqueue_mult.enqueueReadBuffer(device_result,CL_TRUE,0,sizeof(float)*GLOBAL_SIZE,host_result.data());

    for(int i = 0;i<GLOBAL_SIZE;i++)
        cout<< "result : " << host_result[i] <<endl; 
}
