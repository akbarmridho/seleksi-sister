#ifndef LOADER_H
#define LOADER_H

#include "datatypes.h"
#include <string>
#include <opencv2/core.hpp>

size_t load_image(rgb_image_t *dest, const cv::Mat &img_data, int *height, int *width);

cv::Mat resize(const cv::Mat &source, int width, int height);

#endif