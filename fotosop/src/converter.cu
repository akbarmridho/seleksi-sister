#include "converter.cuh"

/**
 * Modified from https://gist.github.com/ciembor/1494530
 * @param r
 * @param g
 * @param b
 * @return
 */
__device__ hsl_t rgb_to_hsl(float r, float g, float b) {
    hsl_t result;

    r /= 255;
    g /= 255;
    b /= 255;

    float val_max = max(max(r, g), b);
    float val_min = min(min(r, g), b);

    result.h = result.s = result.l = (val_max + val_min) / 2;

    if (val_max == val_min) {
        result.h = result.s = 0;
    } else {
        float diff = val_max - val_min;
        result.s = (result.l > 0.5) ? (diff / (2 - val_max - val_min)) : (diff / (val_max + val_min));

        if (val_max == r) {
            result.h = (g - b) / diff + (g < b ? 6.0 : 0.0);
        } else if (val_max == g) {
            result.h = (b - r) / diff + 2;
        } else if (val_max == b) {
            result.h = (r - g) / diff + 4;
        }

        result.h /= 6.0;
    }

    return result;
}

/**
 * modified from https://gist.github.com/ciembor/1494530
 * @param p
 * @param q
 * @param t
 * @return
 */
__device__ float hue_to_rgb(float p, float q, float t) {

    if (t < 0)
        t += 1;
    if (t > 1)
        t -= 1;
    if (t < 1. / 6)
        return p + (q - p) * 6 * t;
    if (t < 1. / 2)
        return q;
    if (t < 2. / 3)
        return p + (q - p) * (2. / 3 - t) * 6;

    return p;
}

/**
 * Modified from https://gist.github.com/ciembor/1494530
 * @param h
 * @param s
 * @param l
 * @return
 */
__device__ rgb_t hsl_to_rgb(float h, float s, float l) {

    rgb_t result;

    if (0 == s) {
        result.r = result.g = result.b = l * 255; // achromatic
    } else {
        float q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        float p = 2 * l - q;
        result.r = hue_to_rgb(p, q, h + 1. / 3) * 255;
        result.g = hue_to_rgb(p, q, h) * 255;
        result.b = hue_to_rgb(p, q, h - 1. / 3) * 255;
    }

    return result;

}


__global__ void kernel_to_grey(rgb_image_t image, int height, int width) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        unsigned int grey_offset = y * width + x;
        unsigned int rgb_offset = grey_offset * CHANNELS;

        uint16_t r = image[rgb_offset + 0];
        uint16_t g = image[rgb_offset + 1];
        uint16_t b = image[rgb_offset + 2];

        auto value = (uint8_t) ((r + g + b) / 3);

        image[rgb_offset + 0] = value;
        image[rgb_offset + 1] = value;
        image[rgb_offset + 2] = value;

    }
}

__global__ void kernel_add_contrast(rgb_image_t image, int height, int width, int value) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        unsigned int pixel_offset = y * width + x;
        unsigned int rgb_offset = pixel_offset * CHANNELS;

        for (int i = 0; i < 3; i++) {
            uint16_t c = image[rgb_offset + i];
            image[rgb_offset + i] = max(0, min(255, value * (c - 128) + 128));
        }
    }
}

__global__ void kernel_add_saturation(rgb_image_t image, int height, int width, float value) {
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        unsigned int pixel_offset = y * width + x;
        unsigned int rgb_offset = pixel_offset * CHANNELS;

        float r = image[rgb_offset + 0];
        float g = image[rgb_offset + 1];
        float b = image[rgb_offset + 2];

        hsl_t hsl = rgb_to_hsl(r, g, b);
        hsl.s *= (1.0 + value);

        rgb_t rgb = hsl_to_rgb(hsl.h, hsl.s, hsl.l);

        image[rgb_offset + 0] = (uint8_t) rgb.r;
        image[rgb_offset + 1] = (uint8_t) rgb.g;
        image[rgb_offset + 2] = (uint8_t) rgb.b;
    }
}

void to_grey(rgb_image_t image, int height, int width) {
    const dim3 dimGrid((int) ceil(width / 16.0), (int) ceil(height / 16.0));
    const dim3 dimBlock(16, 16);

    kernel_to_grey<<<dimGrid, dimBlock>>>(image, height, width);
}


void add_contrast(rgb_image_t image, int height, int width, int value) {
    if (value == 0) {
        return;
    }

    // contrast modifier value range [-255, 255]
    const dim3 dimGrid((int) ceil(width / 16.0), (int) ceil(height / 16.0));
    const dim3 dimBlock(16, 16);

    kernel_add_contrast<<<dimGrid, dimBlock>>>(image, height, width, (int) value);
}

void add_saturation(rgb_image_t image, int height, int width, float value) {
    if (value == 0.0) {
        return;
    }

    // contrast modifier value range [-1, 1]
    const dim3 dimGrid((int) ceil(width / 16.0), (int) ceil(height / 16.0));
    const dim3 dimBlock(16, 16);

    kernel_add_saturation<<<dimGrid, dimBlock>>>(image, height, width, value);
}
