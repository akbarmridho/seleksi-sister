#include "converter.cuh"

__global__ void kernel_to_grey(crgb_image_t source, gray_image_t result, int height, int width) {
    unsigned int x = threadIdx.x + blockIdx.x + blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y + blockDim.y;

    if (x < width && y < height) {
        unsigned int grey_offset = y * width + x;
        unsigned int rgb_offset = grey_offset * CHANNELS;

        uint16_t r = source[rgb_offset + 0];
        uint16_t g = source[rgb_offset + 1];
        uint16_t b = source[rgb_offset + 2];

        result[grey_offset] = (uint8_t) ((r + g + b) / 3);
    }
}

void to_grey(crgb_image_t source, gray_image_t result, int height, int width) {
    const dim3 dimGrid((int) ceil(width / 16.0), (int) ceil(height / 16.0));
    const dim3 dimBlock(16, 16);

    kernel_to_grey<<<dimGrid, dimBlock>>>(source, result, height, width);
}


