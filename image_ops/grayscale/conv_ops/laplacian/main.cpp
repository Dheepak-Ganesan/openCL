#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define kH 5
#define kW 5

int main() {
    // OpenCL setup
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Load image from TXT file
    ifstream file("image_2d.txt");
    if (!file.is_open()) {
        cerr << "Error: Could not open image_2d.txt\n";
        return -1;
    }

    int WIDTH = 0, HEIGHT = 0;
    file >> WIDTH >> HEIGHT;
    cout << "Image dimensions : H " << HEIGHT << " x W " << WIDTH << endl;

    int GLOBAL_DIM = HEIGHT * WIDTH;
    int outH = HEIGHT - kH + 1;
    int outW = WIDTH - kW + 1;

    vector<int> input_image(GLOBAL_DIM);
    vector<int> output_image(outH * outW);
    vector<int> laplacian_filter(kH * kW);

    // Select kernel
    int kernel_size = kH * kW;
    if (kernel_size == 9) { // 3x3 Laplacian
        laplacian_filter = { 0,1,0, 1,-4,1, 0,1,0 };
    } else if (kernel_size == 25) { // 5x5 Laplacian
        laplacian_filter = {
            0,0,-1,0,0,
            0,-1,-2,-1,0,
            -1,-2,16,-2,-1,
            0,-1,-2,-1,0,
            0,0,-1,0,0
        };
    } else if (kernel_size == 49) { // 7x7 Laplacian
        laplacian_filter = {
            0,0,0,-1,0,0,0,
            0,0,-1,-2,-1,0,0,
            0,-1,-2,-4,-2,-1,0,
            -1,-2,-4,24,-4,-2,-1,
            0,-1,-2,-4,-2,-1,0,
            0,0,-1,-2,-1,0,0,
            0,0,0,-1,0,0,0
        };
    } else {
       
    }

    // Fill input image from TXT
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            file >> input_image[row * WIDTH + col];

    
    std::cout << "=== Sample pixels BEFORE kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << input_image[row * WIDTH + col] << "\t";
    }
    std::cout << "\n";
}
    // Compute maximum absolute sum for normalization
    int kernel_abs_sum = 0;
    for(auto &val : laplacian_filter) kernel_abs_sum += abs(val);
    int max_abs_sum = kernel_abs_sum * 255;

    // OpenCL buffers
    cl::Buffer input_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*input_image.size(), input_image.data());
    cl::Buffer filter_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*laplacian_filter.size(), laplacian_filter.data());
    cl::Buffer out_dev(context, CL_MEM_WRITE_ONLY, sizeof(int)*output_image.size());

    // Build program & kernel
    cl::Program program(context, util::loadProgram("laplacian_filter.cl"));
    program.build({device});
    cl::Kernel kernel(program, "laplacian_filter");

    kernel.setArg(0, input_dev);
    kernel.setArg(1, filter_dev);
    kernel.setArg(2, out_dev);
    kernel.setArg(3, kH);
    kernel.setArg(4, kW);
    kernel.setArg(5, outH);
    kernel.setArg(6, outW);
    kernel.setArg(7, HEIGHT);
    kernel.setArg(8, WIDTH);
    kernel.setArg(9, max_abs_sum); 

    // Run kernel
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(outH, outW), cl::NullRange, nullptr, &event);
    event.wait();
    queue.enqueueReadBuffer(out_dev, CL_TRUE, 0, sizeof(int)*output_image.size(), output_image.data());

    std::cout << "=== Sample pixels AFTER kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << output_image[row * outW + col] << "\t";
        }
        std::cout << "\n";
    }
    // Save output
    ofstream out_file("laplacian_norm.txt");
    out_file << outW << " " << outH << "\n";
    for(int row=0; row<outH; row++){
        for(int col=0; col<outW; col++){
            out_file << output_image[row*outW + col];
            if(col < outW-1) out_file << " ";
        }
        out_file << "\n";
    }
    out_file.close();
    cout << "Output saved to laplacian_norm.txt\n";
}

/*
Execution commands:
 g++ main.cpp -o main -lOpenCL -I /home/mcw/Documents/openCL/Exercises-Solutions-1.2.1/Exercises/Cpp_common
 ./main
*/
   
