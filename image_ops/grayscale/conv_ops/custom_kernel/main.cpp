#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#define kH 3 
#define kW 3

int main() {

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Load image
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

    // Fill input image
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
            file >> input_image[row * WIDTH + col];
    file.close();

    vector<int> custom_filter = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    cout << "=== Sample pixels BEFORE kernel ===\n";
    for (int row = 0; row < min(5, HEIGHT); row++) {
        for (int col = 0; col < min(5, WIDTH); col++)
            cout << input_image[row * WIDTH + col] << "\t";
        cout << "\n";
    }

    cl::Buffer input_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*input_image.size(), input_image.data());
    cl::Buffer filter_dev(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*custom_filter.size(), custom_filter.data());
    cl::Buffer out_dev(context, CL_MEM_WRITE_ONLY, sizeof(int)*output_image.size());

    cl::Program program(context, util::loadProgram("custom_filter.cl"));
    program.build({device});
    cl::Kernel kernel(program, "custom_filter");

    kernel.setArg(0, input_dev);
    kernel.setArg(1, filter_dev);
    kernel.setArg(2, out_dev);
    kernel.setArg(3, kH);
    kernel.setArg(4, kW);
    kernel.setArg(5, outH);
    kernel.setArg(6, outW);
    kernel.setArg(7, HEIGHT);
    kernel.setArg(8, WIDTH);

    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(outH, outW), cl::NullRange, nullptr, &event);
    event.wait();

    queue.enqueueReadBuffer(out_dev, CL_TRUE, 0, sizeof(int)*output_image.size(), output_image.data());

    cout << "=== Sample pixels AFTER kernel ===\n";
    for (int row = 0; row < min(5, outH); row++) {
        for (int col = 0; col < min(5, outW); col++)
            cout << output_image[row * outW + col] << "\t";
        cout << "\n";
    }

    ofstream out_file("custom_filter.txt");
    out_file << outW << " " << outH << "\n";
    for(int row=0; row<outH; row++){
        for(int col=0; col<outW; col++){
            out_file << output_image[row*outW + col];
            if(col < outW-1) out_file << " ";
        }
        out_file << "\n";
    }
    out_file.close();

    cout << "Output saved to custom_filter.txt\n";

    return 0;
}
