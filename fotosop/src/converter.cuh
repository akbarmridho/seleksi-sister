#ifndef CONVERTER_H
#define CONVERTER_H

#include "datatypes.cuh"

void to_grey(crgb_image_t source, gray_image_t result, int height, int width);

#endif