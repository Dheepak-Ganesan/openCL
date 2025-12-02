#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
using namespace std;

#define kH 5
#define kW 5

int main() {

    std::vector<cl::Platform> num_platforms;
    cl::Platform::get(&num_platforms);   
    cl::Platform platform = num_platforms[0];

    std::vector<cl::Device> num_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &num_devices);
    cl::Device device = num_devices[0];
    
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Load image matrix from TXT file and check if file opens successfully
    std::ifstream file("image_2d.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open image_2d.txt\n";
        return -1;
    }

    int WIDTH = 0, HEIGHT = 0;
    file >> WIDTH >> HEIGHT;
    cout << "Image dimensions : H " << HEIGHT << " x W " << WIDTH << endl;

    //setting the global dimensions
    int GLOBAL_DIM = HEIGHT * WIDTH;

    //output dimensions
    int outH = HEIGHT - kH + 1; 
    int outW =  WIDTH - kW + 1;

    std::vector<int> input_image(GLOBAL_DIM);
    std::vector<int> output_image(outH * outW);
    std::vector<int> avg_filter(kH * kW,1);

    // Fill vector from txt file
    for(int row = 0; row < HEIGHT; row++)
    {
        for(int col = 0; col < WIDTH; col++)
        {
            int val;
            file >> val;
            input_image[row * WIDTH + col] = val;
        }
    }

    std::cout << "=== Sample pixels BEFORE kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << input_image[row * WIDTH + col] << "\t";
    }
    std::cout << "\n";
}
    cl::Buffer input_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * input_image.size(),input_image.data());
    cl::Buffer filter_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * avg_filter.size(),avg_filter.data());
    cl::Buffer out_dev(context, CL_MEM_WRITE_ONLY,sizeof(int) * output_image.size());

    cl::Program program(context, util::loadProgram("avg_filter.cl"));
    program.build({device});

    cl::Kernel kernel_avg_filter(program, "avg_filter");
    kernel_avg_filter.setArg(0,input_dev);
    kernel_avg_filter.setArg(1, filter_dev);
    kernel_avg_filter.setArg(2,out_dev);
    kernel_avg_filter.setArg(3, kH);
    kernel_avg_filter.setArg(4, kW);
    kernel_avg_filter.setArg(5,outH);
    kernel_avg_filter.setArg(6,outW);
    kernel_avg_filter.setArg(7,HEIGHT);
    kernel_avg_filter.setArg(8,WIDTH);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel_avg_filter,cl::NullRange,cl::NDRange(outH,outW),cl::NullRange,nullptr,&event);
    event.wait();

    queue.enqueueReadBuffer(out_dev, CL_TRUE, 0, sizeof(int) * output_image.size(),output_image.data());

    std::cout << "=== Sample pixels AFTER kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << output_image[row * outW + col] << "\t";
        }
        std::cout << "\n";
    }

    // Open output file
    std::ofstream out_file("avg_filter.txt");
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not create avg_filter.txt\n";
        return -1;
    }

    // Write dimensions first
    out_file << outW << " " << outH << "\n";

    for (int row = 0; row < outH; row++) {
        for (int col = 0; col < outW; col++) {
            out_file << output_image[row * outW + col];
            if (col < outW - 1) out_file << " ";
        }
        out_file << "\n";
    }
    

    out_file.close();
    std::cout << "Output image saved to avg_filter.txt\n";

}

/*
Execution commands:
 g++ main.cpp -o main -lOpenCL -I /home/mcw/Documents/openCL/Exercises-Solutions-1.2.1/Exercises/Cpp_common
 ./main
*/
   

