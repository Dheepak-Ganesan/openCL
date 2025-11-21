#define __ENABLE_CL_EXCEPTIONS
#include <iostream>
#include <vector>
#include "cl.hpp"
#include "util.hpp"
using namespace std;

int main() {
    int W = 4, H = 4, D = 2;
    int total = W * H * D;

    vector<float> arr(total);
    for(int i=0;i<total;i++)
        arr[i] = i;    // 0,1,2,3,...

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer buf(context,
                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                   sizeof(float) * total,
                   arr.data());

    cl::Program program(context, util::loadProgram("3d_kernel.cl"));
    program.build({device});

    cl::Kernel kernel(program, "eltwise_3d");

    kernel.setArg(0, buf);
    kernel.setArg(1, W);
    kernel.setArg(2, H);
    kernel.setArg(3, D);

    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(W, H, D),
        cl::NullRange
    );

    queue.finish();

    queue.enqueueReadBuffer(buf, CL_TRUE, 0,
                            sizeof(float) * total,
                            arr.data());

    cout << "Updated 3D data:\n";
    int idx = 0;
    for (int z = 0; z < D; z++) {
        cout << "Slice z=" << z << "\n";
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                cout << arr[idx++] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }
}
