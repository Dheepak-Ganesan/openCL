#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    std::ifstream file("counter_90deg_rot_3d.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open counter_90deg_rot_3d.txt\n";
        return -1;
    }

    int height, width, channels;
    file >> channels >> height >> width;
    std::cout << "Image dimensions: " << width << " x " << height << " x " << channels << std::endl;

    std::vector<int> pixels(channels * height * width);

    for (int c = 0; c < channels; c++)
        for (int h = 0; h < height; h++)
            for (int w = 0; w < width; w++) {
                int val;
                file >> val;
                pixels[c * height * width + h * width + w] = val;
            }
    file.close();

    cv::Mat img(height, width, CV_8UC3);

    for (int h = 0; h < height; h++)
        for (int w = 0; w < width; w++) {
            cv::Vec3b &pixel = img.at<cv::Vec3b>(h, w);
            for (int c = 0; c < channels; c++)
                pixel[c] = static_cast<uchar>(pixels[c * height * width + h * width + w]);
        }

    cv::imwrite("counter_90deg_rot_3d.png", img);
    std::cout << "Image saved as counter_90deg_rot_3d.png\n";
    return 0;
}

/*
Compile and run:
g++ outmatrix_to_image.cpp -o outmatrix_to_image `pkg-config --cflags --libs opencv4`
./outmatrix_to_image
*/
