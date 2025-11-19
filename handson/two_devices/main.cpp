#include "cl.hpp"
#include <iostream>
#include <vector>
#include "util.hpp"

#define N 512

// // Simple kernel: C[i] = A[i] + B[i]
// const char* KERNEL_SRC = R"(
// __kernel void eltwise_add(__global int* A,
//                           __global int* B,
//                           __global int* C,
//                           const int offset)
// {
//     int id = get_global_id(0) + offset;  
//     C[id] = A[id] + B[id];
// })";

int main()
{
    // ------------------------------------------------------------------
    // 1. Get platforms and devices
    // ------------------------------------------------------------------
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cout << "No OpenCL platforms found.\n";
        return 1;
    }

    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        std::cout << "No OpenCL devices found.\n";
        return 1;
    }

    // Use at least one device, at most two
    cl::Device dev0 = devices[0];
    cl::Device dev1 = (devices.size() > 1 ? devices[1] : devices[0]);
    // cl::Device dev1=devices[1]; //results in Segmentation fault (core dumped) since the system has only one device

    std::cout << "Devices found: " << devices.size() << std::endl;
for (size_t i = 0; i < devices.size(); ++i)
    std::cout << "  Device " << i << ": " 
              << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;

    // ------------------------------------------------------------------
    // 2. Create two contexts and two queues
    // ------------------------------------------------------------------
    cl::Context ctx0(dev0);
    cl::Context ctx1(dev1);

    cl::CommandQueue q0(ctx0, dev0);
    cl::CommandQueue q1(ctx1, dev1);

    // ------------------------------------------------------------------
    // 3. Prepare host data
    // ------------------------------------------------------------------
    std::vector<int> A(N, 50);
    std::vector<int> B(N, 20);
    std::vector<int> OUT(N, 0);

    // ------------------------------------------------------------------
    // 4. Create buffers on both devices
    // ------------------------------------------------------------------
    cl::Buffer A0(ctx0, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(int)*N, A.data());
    cl::Buffer B0(ctx0, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(int)*N, B.data());
    cl::Buffer C0(ctx0, CL_MEM_WRITE_ONLY, sizeof(int)*N);

    cl::Buffer A1(ctx1, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(int)*N, A.data());
    cl::Buffer B1(ctx1, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, sizeof(int)*N, B.data());
    cl::Buffer C1(ctx1, CL_MEM_WRITE_ONLY, sizeof(int)*N);

    // ------------------------------------------------------------------
    // 5. Build program on each device
    // ------------------------------------------------------------------
    cl::Program program0(ctx0,util::loadProgram("compute.cl"));
    cl::Program program1(ctx1,util::loadProgram("compute.cl"));

    program0.build({dev0});
    program1.build({dev1});

    // ------------------------------------------------------------------
    // 6. Create kernels
    // ------------------------------------------------------------------
    cl::Kernel k0(program0, "compute");
    cl::Kernel k1(program1, "compute");

    // Each device handles half of the array
    int half = N / 2;

    // ------------------------------------------------------------------
    // 7. Set arguments for device 0 (first half)
    // ------------------------------------------------------------------
    k0.setArg(0, A0);
    k0.setArg(1, B0);
    k0.setArg(2, C0);
    k0.setArg(3, 0);      // offset = 0 for first half

    // ------------------------------------------------------------------
    // 8. Set arguments for device 1 (second half)
    // ------------------------------------------------------------------
    k1.setArg(0, A1);
    k1.setArg(1, B1);
    k1.setArg(2, C1);
    k1.setArg(3, half);   // offset = 256

    // ------------------------------------------------------------------
    // 9. Enqueue kernels
    // ------------------------------------------------------------------
    cl::NDRange global(half);

    q0.enqueueNDRangeKernel(k0, cl::NullRange, global);
    q1.enqueueNDRangeKernel(k1, cl::NullRange, global);

    q0.finish();
    q1.finish();

    // ------------------------------------------------------------------
    // 10. Read back results into correct halves of OUT[]
    // ------------------------------------------------------------------
    q0.enqueueReadBuffer(C0, CL_TRUE,
                         0,
                         sizeof(int)*half,
                         OUT.data());

    q1.enqueueReadBuffer(C1, CL_TRUE,
                         sizeof(int)*half,
                         sizeof(int)*half,
                         OUT.data() + half);

    // ------------------------------------------------------------------
    // 11. Print
    // ------------------------------------------------------------------
    for (int i = 0; i < N; i++)
        std::cout << "OUT[" << i << "] = " << OUT[i] << std::endl;

    return 0;
}
