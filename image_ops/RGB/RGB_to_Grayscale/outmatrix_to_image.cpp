#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    std::ifstream file("output_matrix.txt");
    if (!file.is_open()) return -1;

    int channels, height, width;
    file >> channels >> height >> width;

    std::vector<int> pixels(width * height);

    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
            int val;
            file >> val;
            pixels[row * width + col] = val;
        }

    file.close();

    cv::Mat img(height, width, CV_8UC1);

    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            img.at<uchar>(row, col) = static_cast<uchar>(pixels[row * width + col]);

    cv::imwrite("grayscale.png", img);

    return 0;
}

/*
Compile and run:
g++ outmatrix_to_image.cpp -o outmatrix_to_image `pkg-config --cflags --libs opencv4`
./outmatrix_to_image
*/