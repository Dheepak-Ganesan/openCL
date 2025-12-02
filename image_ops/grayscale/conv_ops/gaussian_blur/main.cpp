#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define kH 5
#define kW 5

int main() {
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
    vector<int> gaussian_filter(kH * kW);

    // Choose kernel and divisor based on size
    int kernel_size = kH * kW;
    int divisor = 1;

    if (kernel_size == 9) {
        gaussian_filter = {
            1, 2, 1,
            2, 4, 2,
            1, 2, 1
        };
        divisor = 16;
    } 
    else if (kernel_size == 25) {
        gaussian_filter = {
            1,  4,  6,  4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1,  4,  6,  4, 1
        };
        divisor = 256;
    } 
    else if (kernel_size == 49) {
        gaussian_filter = {
            1,   4,   7,  10,  7,  4, 1,
            4,  12,  26,  33, 26, 12, 4,
            7,  26,  55,  71, 55, 26, 7,
            10, 33,  71,  91, 71, 33, 10,
            7,  26,  55,  71, 55, 26, 7,
            4,  12,  26,  33, 26, 12, 4,
            1,   4,   7,  10,  7,  4, 1
        };
        divisor = 1115;
    } 
    else {
        cout << "Only 3x3, 5x5, 7x7 kernels are supported in this version\n";
    }

    // Fill input image from TXT
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            int val;
            file >> val;
            input_image[row * WIDTH + col] = val;
        }
    }

    cout << "=== Sample pixels BEFORE kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            cout << input_image[row * WIDTH + col] << "\t";
        }
        cout << "\n";
    }

    // Allocate OpenCL buffers
    cl::Buffer input_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * input_image.size(), input_image.data());
    cl::Buffer filter_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * gaussian_filter.size(), gaussian_filter.data());
    cl::Buffer out_dev(context, CL_MEM_WRITE_ONLY, sizeof(int) * output_image.size());

    // Build program and set kernel
    cl::Program program(context, util::loadProgram("gaussian_filter.cl"));
    program.build({device});

    cl::Kernel kernel(program, "gaussian_filter");
    kernel.setArg(0, input_dev);
    kernel.setArg(1, filter_dev);
    kernel.setArg(2, out_dev);
    kernel.setArg(3, kH);
    kernel.setArg(4, kW);
    kernel.setArg(5, outH);
    kernel.setArg(6, outW);
    kernel.setArg(7, HEIGHT);
    kernel.setArg(8, WIDTH);
    kernel.setArg(9, divisor);

    // Execute kernel
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(outH, outW), cl::NullRange, nullptr, &event);
    event.wait();

    queue.enqueueReadBuffer(out_dev, CL_TRUE, 0, sizeof(int) * output_image.size(), output_image.data());

    cout << "=== Sample pixels AFTER kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            cout << output_image[row * outW + col] << "\t";
        }
        cout << "\n";
    }

    // Save output image to TXT
    ofstream out_file("gaussian_filter.txt");
    if (!out_file.is_open()) {
        cerr << "Error: Could not create gaussian_filter.txt\n";
        return -1;
    }

    out_file << outW << " " << outH << "\n";
    for (int row = 0; row < outH; row++) {
        for (int col = 0; col < outW; col++) {
            out_file << output_image[row * outW + col];
            if (col < outW - 1) out_file << " ";
        }
        out_file << "\n";
    }
    out_file.close();

    cout << "Output image saved to gaussian_filter.txt\n";
}

/*
Execution commands:
g++ main.cpp -o main -lOpenCL -I /home/mcw/Documents/openCL/Exercises-Solutions-1.2.1/Exercises/Cpp_common
./main

g++ outmatrix_to_image.cpp -o outmatrix_to_image `pkg-config --cflags --libs opencv4`
./outmatrix_to_image
*/
   

