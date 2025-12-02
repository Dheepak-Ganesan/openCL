#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define _ENABLE_CL_EXCEPTIONS

int main()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);   
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];
    
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    std::ifstream file("image_3d.txt");
    if (!file.is_open()) return -1;

    int HEIGHT = 0, WIDTH = 0, CHANNELS = 0;
    file >> CHANNELS >> HEIGHT >> WIDTH ;

    int GLOBAL_DIM = HEIGHT * WIDTH * CHANNELS;
    std::vector<int> input_image(GLOBAL_DIM);

    for(int c = 0; c < CHANNELS; c++)
        for(int h = 0; h < HEIGHT; h++)
            for(int w = 0; w < WIDTH; w++)
            {
                int val;
                file >> val;
                input_image[c * HEIGHT * WIDTH + h * WIDTH + w] = val;
            }
    file.close();

    cl::Buffer device_image(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * GLOBAL_DIM, input_image.data());

    std::string kernel_src = util::loadProgram("neg_img.cl");
    cl::Program program(context, kernel_src);
    program.build({device});

    cl::Kernel kernel(program, "neg_img");
    kernel.setArg(0, device_image);
    kernel.setArg(1, HEIGHT);
    kernel.setArg(2, WIDTH);
    kernel.setArg(3, CHANNELS);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(HEIGHT, WIDTH, CHANNELS));
    queue.finish();

    queue.enqueueReadBuffer(device_image, CL_TRUE, 0, sizeof(int) * GLOBAL_DIM, input_image.data());

    std::ofstream out_file("negative_image_3d.txt");
    out_file << CHANNELS << " " << HEIGHT << " " << WIDTH << "\n";
    for(int c = 0; c < CHANNELS; c++)
    {
        for(int h = 0; h < HEIGHT; h++)
        {
            for(int w = 0; w < WIDTH; w++)
            {
                out_file << input_image[c * HEIGHT * WIDTH + h * WIDTH + w];
                if(w < WIDTH - 1) out_file << " ";
            }
            out_file << "\n";
        }
        out_file << "\n";
    }
    out_file.close();

    std::cout << "Output 3D negative image saved to negative_image_3d.txt\n";
}
