#include "cl.hpp"
#include "util.hpp"
#include <iostream>
#include <vector>
using namespace std;

#define H 6
#define W 6
#define D 6
#define kH 3
#define kW 3
#define kD 3
#define oH (H- kH + 1)
#define oW (W - kW + 1)
#define oD (D - kD + 1)
#define WG 2
#define K  3
#define patch WG+K-1

int main()
{
    vector<cl::Platform> platforms_list;
    cl::Platform::get(&platforms_list);
    cl::Platform platform = platforms_list[0];

    vector<cl::Device> devices_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL,&devices_list);
    cl::Device device = devices_list[0];

    cl::Context context(device);
    cl::CommandQueue queue(context,device);

    vector<int> input(H * W * D);
    for(int i = 0 ; i< D; i++) //depth 
    {
        for(int j = 0; j < H; j++) //height
        {
            for(int k = 0; k < W; k++) //width
            {
                // 2d flatten to 1d - i * width +j
                // 3d flatten to 1d - i*(H*W) + j*width + k
                input[(i *(H*W)) + (j*W) + k] = i+j+k;
            }
        }
    }

    vector<int> filter(kD * kH * kW,1);
    vector<int> output(oD * oH * oW,0);

    cl::Buffer input_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * input.size(),input.data());
    cl::Buffer filter_dev(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * filter.size(),filter.data());
    cl::Buffer out_dev(context,CL_MEM_READ_WRITE,sizeof(int) * output.size());

    cl::Program program(context,util::loadProgram("conv3d_local.cl"));
    program.build({device});

    cl::Kernel kernel(program,"conv3d_local");
    kernel.setArg(0,input_dev);
    kernel.setArg(1,filter_dev);
    kernel.setArg(2,out_dev);
    kernel.setArg(3,D);
    kernel.setArg(4,H);
    kernel.setArg(5,W);
    kernel.setArg(6,kD);
    kernel.setArg(7,kH);
    kernel.setArg(8,kW);
    kernel.setArg(9,oD);
    kernel.setArg(10,oH);
    kernel.setArg(11,oW);
    kernel.setArg(12,WG);
    kernel.setArg(13,patch);

    int gD = ((oD+WG-1)/WG) * WG;
    int gH = ((oH+WG-1)/WG) * WG;
    int gW = ((oW+WG-1)/WG) * WG;

    //global size needs to be a multiple of local size, so instead of putting oD,oH,oW, (oD+WG-1/WG) / WG,...
    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(gD,gH,gW),cl::NDRange(WG,WG,WG));
    queue.enqueueReadBuffer(out_dev,CL_TRUE,0,sizeof(int)*output.size(),output.data());

    cout << "3D convolution Results using local memory" <<endl;
    for(int i = 0 ; i< oD; i++) //depth 
    {
        for(int j = 0; j < oH; j++) //height
        {
            for(int k = 0; k < oW; k++) //width
            {
                
               cout << output[(i *(oH*oW)) + (j*oW) + k] <<" ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
   
}