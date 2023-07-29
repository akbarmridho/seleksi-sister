#ifndef CONVERTER_H
#define CONVERTER_H

#include "datatypes.h"

void to_grey(rgb_image_t image, int height, int width);

void add_contrast(rgb_image_t image, int height, int width, int value);

void add_saturation(rgb_image_t image, int height, int width, float value);

#endif