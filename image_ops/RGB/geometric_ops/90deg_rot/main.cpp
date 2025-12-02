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
    if (!file.is_open()) return -1;

    int HEIGHT=0, WIDTH=0, CHANNELS=0;
    file >> CHANNELS >> HEIGHT >> WIDTH;

    int GLOBAL_DIM = HEIGHT * WIDTH * CHANNELS;
    std::vector<int> input_image(GLOBAL_DIM);
    std::vector<int> output_image(GLOBAL_DIM);

    for(int c=0; c<CHANNELS; c++)
        for(int h=0; h<HEIGHT; h++)
            for(int w=0; w<WIDTH; w++)
            {
                int val;
                file >> val;
                input_image[c * HEIGHT * WIDTH + h * WIDTH + w] = val;
            }

    file.close();

    cout << "=== Sample pixels BEFORE kernel (Red channel) ===\n";
    for(int row=0; row<5; row++)
    {
        for(int col=0; col<5; col++)
            cout << input_image[0 * WIDTH * HEIGHT + row * WIDTH + col] << "\t";
        cout << "\n";
    }

    cl::Buffer device_image(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * GLOBAL_DIM, input_image.data());
    cl::Buffer device_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * GLOBAL_DIM);

    std::string kernel_source = util::loadProgram("90deg_rot.cl");
    cl::Program program_obj(context, kernel_source);
    program_obj.build({device});

    cl::Kernel kernel_90deg_rot(program_obj, "rotate_90_clkWise");
    kernel_90deg_rot.setArg(0, device_image);
    kernel_90deg_rot.setArg(1, device_output);
    kernel_90deg_rot.setArg(2, HEIGHT);
    kernel_90deg_rot.setArg(3, WIDTH);
    kernel_90deg_rot.setArg(4, CHANNELS);

    CommandQueue.enqueueNDRangeKernel(kernel_90deg_rot, cl::NullRange, cl::NDRange(HEIGHT, WIDTH, CHANNELS));
    CommandQueue.finish();

    CommandQueue.enqueueReadBuffer(device_output, CL_TRUE, 0, sizeof(int) * GLOBAL_DIM, output_image.data());

    cout << "=== Sample pixels AFTER kernel (Red channel) ===\n";
    for(int row=0; row<5; row++)
    {
        for(int col=0; col<5; col++)
            cout << output_image[0 * WIDTH * HEIGHT + row * WIDTH + col] << "\t";
        cout << "\n";
    }

    ofstream out_file("90deg_rot_3d.txt");
    if(!out_file.is_open()) return -1;
    out_file << CHANNELS << " " << HEIGHT << " " << WIDTH << "\n";

    for (int c = 0; c < CHANNELS; c++) {
        for (int h = 0; h < HEIGHT; h++) {
            for (int w = 0; w < WIDTH; w++) {
                out_file << output_image[c * (HEIGHT * WIDTH) + h * WIDTH + w];
                if (w < WIDTH - 1) out_file << " ";
            }
            out_file << "\n";
        }
        out_file << "\n"; 
    }

    out_file.close();
    cout << "Output image saved to 90deg_rot_3d.txt\n";
}
