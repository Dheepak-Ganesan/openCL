#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define _ENABLE_CL_EXCEPTIONS

int main()
{
    std::vector<cl::Platform> num_platforms;
    cl::Platform::get(&num_platforms);   
    cl::Platform platform = num_platforms[0];

    std::vector<cl::Device> num_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &num_devices);
    cl::Device device = num_devices[0];
    
    cl::Context context(device);
    cl::CommandQueue CommandQueue(context, device);

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

    std::vector<int> input_image(GLOBAL_DIM);
    std::vector<int> output_image(GLOBAL_DIM);

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
    cl::Buffer device_image(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * GLOBAL_DIM, input_image.data());
    cl::Buffer device_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * GLOBAL_DIM);

    std::string kernel_source = util::loadProgram("vertical_flip.cl");
    cl::Program program_obj(context, kernel_source);
    program_obj.build({device});

    cl::Kernel kernel_vertical_flip(program_obj, "vertical_flip");
    kernel_vertical_flip.setArg(0, device_image);
    kernel_vertical_flip.setArg(1,device_output);
    kernel_vertical_flip.setArg(2, WIDTH);
    kernel_vertical_flip.setArg(3,HEIGHT);

    CommandQueue.enqueueNDRangeKernel(kernel_vertical_flip, cl::NullRange, cl::NDRange(HEIGHT,WIDTH));
    CommandQueue.finish();

    CommandQueue.enqueueReadBuffer(device_output, CL_TRUE, 0, sizeof(int) * GLOBAL_DIM, output_image.data());

    std::cout << "=== Sample pixels AFTER kernel ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << output_image[row * WIDTH + col] << "\t";
        }
        std::cout << "\n";
    }

    // Open output file
    std::ofstream out_file("vertical_flip.txt");
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not create vertical_flip.txt\n";
        return -1;
    }

    // Write dimensions first
    out_file << WIDTH << " " << HEIGHT << "\n";

    // Write pixel values row by row
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            out_file << output_image[row * WIDTH + col];
            if (col < WIDTH - 1)
                out_file <<  " ";
        }
        out_file << "\n";
    }

    out_file.close();
    std::cout << "Output image saved to vertical_flip.txt\n";

}

/*
Execution commands:
 g++ main.cpp -o main -lOpenCL -I /home/mcw/Documents/openCL/Exercises-Solutions-1.2.1/Exercises/Cpp_common
 ./main
*/