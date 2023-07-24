#include "loader.cuh"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

size_t load_image(rgb_image_t *dest, const std::string &input_file, int *height, int *width) {
    cv::Mat img_data;

    // read image data
    img_data = cv::imread(input_file, cv::IMREAD_COLOR);

    if (img_data.empty()) {
        std::cerr << "Unable to laod image file: " << input_file << std::endl;
    }

    *height = img_data.rows;
    *width = img_data.cols;

    *dest = (rgb_image_t) malloc(*height * *width * sizeof(uint8_t) * CHANNELS);

    auto rgb_image = (uint8_t *) img_data.data;

    // populate host rgb data array
    for (int i = 0; i < *height * *width * 3; i++) {
        *dest[i] = rgb_image[i];
    }

    return *width * *height;
}

void save_image(const std::string &output_file, uint8_t *result, int height, int width, int type) {
    cv::Mat cv_result(height, width, type, (void *) result);
    cv::imwrite(output_file, cv_result);
}