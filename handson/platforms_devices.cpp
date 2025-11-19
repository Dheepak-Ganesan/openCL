#define __CL_ENABLE_EXCEPTIONS
// step 1. CPP headers and openCL related headers(util,cl.hpp)
#include "cl.hpp"
#include<iostream>
#include<string.h>
#include"util.hpp"
#include<vector>
#include<cstdlib>
#include<cstdio>
#include<fstream> 
// fstream is a library class used for file input and output.
// It allows your program to read from files and write to files. It’s part of the <fstream> header.

#define GLOBAL_LENGTH 1024 // step 2 : set the work items count

// // step 3: set the CL device
// #ifndef DEVICE
// #define DEVICE CL_DEVICE_TYPE_DEFAULT
// #endif

int main()
{

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];


    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&devices);
    
    cl::Device device = devices[0];
    std::cout << platforms.size() << std::endl;
    
}