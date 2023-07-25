#ifndef CONVERTER_H
#define CONVERTER_H

#include "datatypes.cuh"

void to_grey(crgb_image_t source, gray_image_t result, int height, int width);

void add_contrast(crgb_image_t source, rgb_image_t result, int height, int width, float value);

void add_saturation(crgb_image_t source, rgb_image_t result, int height, int width, float value);

#endif