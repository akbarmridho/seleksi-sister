#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

size_t loadImageFile(uint8_t *h_rgb_image, const std::string &input_file, int *height, int *width)
{
    cv::Mat img_data;

    // read image data
    img_data = cv::imread(input_file, cv::IMREAD_COLOR);

    if (img_data.empty())
    {
        std::cerr << "Unable to laod image file: " << input_file << std::endl;
    }

    *height = img_data.rows;
    *width = img_data.cols;

    h_rgb_image = (uint8_t *) malloc(*height * *width * sizeof(uint8_t) * 3);

    auto * rgb_image = (uint8_t*) img_data.data;

    // populate host rgb data array
    for (int i = 0; i < *height * *width * 3; i++)
    {
        h_rgb_image[i] = rgb_image[i];
    }

    return *width * *height;
}

void outputImage(const std::string& output_file, uint8_t* image_result, int height, int width)
{
    cv::Mat result(height, width, CV_8UC1, (void*) image_result);

    cv::imwrite(output_file, result);
}