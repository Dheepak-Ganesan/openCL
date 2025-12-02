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

    std::ifstream file("image_3d.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open image_3d.txt\n";
        return -1;
    }

    int WIDTH = 0, HEIGHT = 0, CHANNELS = 0;
    file >> CHANNELS >> HEIGHT >> WIDTH;
    cout << "Image dimensions : H " << HEIGHT << " x W " << WIDTH << " x C " << CHANNELS << endl;

    int GLOBAL_DIM = WIDTH * HEIGHT * CHANNELS;
    std::vector<int> input_image(GLOBAL_DIM);

    // Fill vector from txt file (CHW order)
    for(int c = 0; c < CHANNELS; c++)
        for(int row = 0; row < HEIGHT; row++)
            for(int col = 0; col < WIDTH; col++)
            {
                int val;
                file >> val;
                input_image[c * WIDTH * HEIGHT + row * WIDTH + col] = val;
            }
    file.close();

    std::cout << "=== Sample pixels BEFORE kernel (Red channel) ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << input_image[0 * WIDTH * HEIGHT + row * WIDTH + col] << "\t";
        }
        std::cout << "\n";
    }

    cl::Buffer device_image(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * GLOBAL_DIM, input_image.data());

    std::string kernel_source = util::loadProgram("threshold.cl");
    cl::Program program_obj(context, kernel_source);
    program_obj.build({device});

    cl::Kernel kernel_add(program_obj, "threshold");
    kernel_add.setArg(0, device_image);
    kernel_add.setArg(1, WIDTH);
    kernel_add.setArg(2, HEIGHT);
    kernel_add.setArg(3, CHANNELS); 

    //global size = WIDTH x HEIGHT x CHANNELS
    CommandQueue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(HEIGHT, WIDTH, CHANNELS));
    CommandQueue.finish();

    CommandQueue.enqueueReadBuffer(device_image, CL_TRUE, 0, sizeof(int) * GLOBAL_DIM, input_image.data());

    std::cout << "=== Sample pixels AFTER kernel (Red channel) ===\n";
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            std::cout << input_image[0 * WIDTH * HEIGHT + row * WIDTH + col] << "\t";
        }
        std::cout << "\n";
    }

    std::ofstream out_file("threshold_image_3d.txt");
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not create threshold_image_3d.txt\n";
        return -1;
    }

    out_file << CHANNELS << " " << HEIGHT << " " << WIDTH << "\n";

    for(int c = 0; c < CHANNELS; c++) {
        for(int row = 0; row < HEIGHT; row++) {
            for(int col = 0; col < WIDTH; col++) {
                out_file << input_image[c * WIDTH * HEIGHT + row * WIDTH + col];
                if(col < WIDTH - 1) out_file << " ";
            }
            out_file << "\n";
        }
        out_file << "\n"; 
    }

    out_file.close();
    std::cout << "Output 3D image saved to threshold_image_3d.txt\n";
}

