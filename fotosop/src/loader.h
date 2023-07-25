#ifndef LOADER_H
#define LOADER_H

#include "datatypes.h"
#include <string>
#include <opencv2/core.hpp>

size_t load_image(rgb_image_t *dest, const std::string &input_file, int *height, int *width);

void save_image(const std::string &output_file, uint8_t *result, int height, int width, int type = CV_8UC3);

#endif