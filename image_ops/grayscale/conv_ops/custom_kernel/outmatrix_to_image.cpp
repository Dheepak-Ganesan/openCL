#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    std::ifstream file("custom_filter.txt");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open custom_filter.txt\n";
        return -1;
    }

    int width, height;
    file >> width >> height;
    std::cout << "Image dimensions: " << width << " x " << height << std::endl;

    std::vector<int> pixels(width * height);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int val;
            file >> val;
            pixels[row * width + col] = val;
        }
    }
    file.close();

    // Create a grayscale OpenCV image
    cv::Mat img(height, width, CV_8UC1);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            img.at<uchar>(row, col) = static_cast<uchar>(pixels[row * width + col]);
        }
    }

    // Save the image
    cv::imwrite("custom_filter.png", img);

    std::cout << "Image saved as custom_filter.png\n";
    return 0;
}

/*
Execution:
g++ outmatrix_to_image.cpp -o outmatrix_to_image `pkg-config --cflags --libs opencv4`
./outmatrix_to_image
*/
