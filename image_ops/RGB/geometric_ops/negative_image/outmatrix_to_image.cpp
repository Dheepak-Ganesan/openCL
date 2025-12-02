#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    std::ifstream file("negative_image_3d.txt");
    if (!file.is_open()) return -1;

    int width, height, channels;
    file >> channels >> height >> width;

    std::vector<int> pixels(width * height * channels);

    for (int c = 0; c < channels; c++)
        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
            {
                int val;
                file >> val;
                pixels[c * width * height + row * width + col] = val;
            }
    file.close();

    cv::Mat img(height, width, CV_8UC3);

    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
            cv::Vec3b& pixel = img.at<cv::Vec3b>(row, col);
            for (int c = 0; c < channels; c++)
                pixel[c] = static_cast<uchar>(pixels[c * width * height + row * width + col]);
        }

    cv::imwrite("negative_image_3d.png", img);

    return 0;
}

/*
Compile and run:
g++ outmatrix_to_image.cpp -o outmatrix_to_image `pkg-config --cflags --libs opencv4`
./outmatrix_to_image
*/
