#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
using namespace std;

int main() 
{
    
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device,CL_QUEUE_PROFILING_ENABLE);

    const int HEIGHT = 512;
    const int WIDTH  = 512;

    // A is HEIGHT × WIDTH  
    // B is WIDTH × HEIGHT  
    // C is HEIGHT × HEIGHT (result)

    vector<int> A(HEIGHT * WIDTH);
    vector<int> B(WIDTH * HEIGHT);
    vector<int> C(HEIGHT * HEIGHT, 0);

    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            A[i*WIDTH + j] = 1;        
        }
    }

    for(int i = 0; i < WIDTH; i++) {
        for(int j = 0; j < HEIGHT; j++) {
            B[i*HEIGHT + j] = 1;        
        }
    }

    cout << "Matrices initialized." << endl;

    cl::Buffer A_buf(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * A.size(), A.data());

    cl::Buffer B_buf(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * B.size(), B.data());

    cl::Buffer C_buf(context, CL_MEM_READ_WRITE,
                     sizeof(int) * C.size());

    cl::Program program(context, util::loadProgram("priv_matmul.cl"));
    program.build({device});

    cl::Kernel kernel(program, "priv_matmul");

    kernel.setArg(0,A_buf);
    kernel.setArg(1,B_buf);
    kernel.setArg(2,C_buf);
    kernel.setArg(3,WIDTH);
    kernel.setArg(4,HEIGHT);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(HEIGHT,HEIGHT),cl::NullRange,nullptr,&event);
    event.wait();

    cl_ulong st = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    double timetaken = (end - st) * 1e-9;
    queue.enqueueReadBuffer(C_buf,CL_TRUE,0,sizeof(int) * HEIGHT * HEIGHT,C.data());

    for(int i=0;i<HEIGHT;i++)
    {
        for(int j=0;j<HEIGHT;j++)
            cout << C[i*HEIGHT + j] << " ";
        cout << endl;
    }
    
    
    // CPU reference multiplication
    vector<int> C_cpu(HEIGHT * HEIGHT, 0);

    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < HEIGHT; col++) {
            int sum = 0;
            for (int k = 0; k < WIDTH; k++) {
                sum += A[row * WIDTH + k] * B[k * HEIGHT + col];
            }
            C_cpu[row * HEIGHT + col] = sum;
        }
    }

    //CPU results
    cout << "\nCPU Matrix Multiplication Result:" << endl;
    for(int i=0; i<HEIGHT; i++) {
        for(int j=0; j<HEIGHT; j++)
            cout << C_cpu[i*HEIGHT + j] << " ";
        cout << endl;
    }

    // Verify GPU vs CPU results
    bool match = true;
    for(int i = 0; i < HEIGHT * HEIGHT; i++) {
        if(C[i] != C_cpu[i]) {
            match = false;
            cout << "Mismatch at index " << i 
                << ": GPU=" << C[i] << ", CPU=" << C_cpu[i] << endl;
            break; // stop at first mismatch
        }
    }

    if(match) {
        cout << "\nGPU and CPU results match perfectly!" << endl;
    } else {
        cout << "\nGPU and CPU results do NOT match!" << endl;
    }

    cout << "Total time taken: " << timetaken << endl;

    int total_flops = 2*HEIGHT*WIDTH*HEIGHT; //2-> multiplication + addition
    double mflops = total_flops / (timetaken * 1000000.0);

    cout << "Total million flops per second: " << mflops <<endl;
}
