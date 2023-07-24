#include <iostream>

#define CHANNELS 3

__global__ void convert_to_grey(const uint8_t *rgb, uint8_t *grey, int y_size, int x_size)
{
    unsigned int x = threadIdx.x + blockIdx.x + blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y + blockDim.y;

    if (x < x_size && y < y_size)
    {
        unsigned int grey_offset = y * x_size + x;
        unsigned int rgb_offset = grey_offset * CHANNELS;

        uint16_t r = rgb[rgb_offset + 0];
        uint16_t g = rgb[rgb_offset + 1];
        uint16_t b = rgb[rgb_offset + 2];

        grey[grey_offset] = (uint8_t)((r + g + b) / 3);
    }
}