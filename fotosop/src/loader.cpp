#include "loader.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

size_t load_image(rgb_image_t *dest, const cv::Mat &img_data, int *height, int *width) {
    *height = img_data.rows;
    *width = img_data.cols;

    *dest = (rgb_image_t) malloc(*height * *width * sizeof(uint8_t) * CHANNELS);

    auto rgb_image = (uint8_t *) img_data.data;

    // populate host rgb data array
    for (int i = 0; i < *height * *width * CHANNELS; i++) {
        (*dest)[i] = rgb_image[i];
    }

    return *width * *height;
}

void save_image(const std::string &output_file, uint8_t *result, int height, int width, int type) {
    cv::Mat cv_result(height, width, type, (void *) result);
    cv::imwrite(output_file, cv_result);
}