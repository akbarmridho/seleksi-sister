#include "loader.h"
#include <opencv2/imgproc/imgproc.hpp>
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

cv::Mat resize(const cv::Mat &source, int width, int height) {
    int original_width = source.cols;
    int original_height = source.rows;

    float ratio = MIN(height / (1.0 * original_height), width / (1.0 * original_width));

    cv::Mat result;
    cv::resize(source, result, cv::Size(int(original_width * ratio), int(original_height * ratio)), cv::INTER_CUBIC);
    return result;
}