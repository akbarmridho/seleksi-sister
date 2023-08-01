#include "blur.cuh"

#include <iostream>

__global__ void kernel_apply_filter(const uint8_t *input,
                                    uint8_t *output,
                                    int height,
                                    int width,
                                    const float *filter,
                                    const int filter_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float r_val = 0.0f;
    float g_val = 0.0f;
    float b_val = 0.0f;

    for (int fx = 0; fx < filter_width; fx++) {
        for (int fy = 0; fy < filter_width; fy++) {
            int pixel_x = x + fx - (filter_width / 2);
            int pixel_y = y + fy - (filter_width / 2);

            pixel_x = min(max(pixel_x, 0), width - 1);
            pixel_y = min(max(pixel_y, 0), height - 1);

            r_val += filter[fy * filter_width + fx] *
                     ((float) input[(pixel_y * width + pixel_x) * CHANNELS + 0]);
            g_val += filter[fy * filter_width + fx] *
                     ((float) input[(pixel_y * width + pixel_x) * CHANNELS + 1]);
            b_val += filter[fy * filter_width + fx] *
                     ((float) input[(pixel_y * width + pixel_x) * CHANNELS + 2]);
        }
    }

    output[(y * width + x) * CHANNELS + 0] = r_val;
    output[(y * width + x) * CHANNELS + 1] = g_val;
    output[(y * width + x) * CHANNELS + 2] = b_val;
}

void generate_gaussian_blur_filter(float **filter, int *filter_width) {
    float sigma = 2;
    *filter_width = 5;
    *filter = new float[*filter_width * *filter_width];

    float mean = (*filter_width) / 2;
    float sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < (*filter_width); ++x)
        for (int y = 0; y < (*filter_width); ++y) {
            (*filter)[x * (*filter_width) + y] =
                    exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
                    / (2 * M_PI * sigma * sigma);

            // Accumulate the kernel values
            sum += (*filter)[x * (*filter_width) + y];
        }

    // Normalize the kernel
    for (int x = 0; x < *filter_width; ++x)
        for (int y = 0; y < *filter_width; ++y)
            // nilai normalization diganti dari
            // (*filter)[x * (*filter_width) + y] /= (sum);
            // menjadi
            (*filter)[x * (*filter_width) + y] /= (sum / 5.3);
    // karena gambar menjadi lebih gelap, entah karena apa :(
}

void gaussian_blur(rgb_image_t image, int height, int width) {
    float *h_filter;
    int filter_width;
    generate_gaussian_blur_filter(&h_filter, &filter_width);

    float *d_filter;
    cudaMalloc(&d_filter, filter_width * filter_width);
    cudaMemcpy(d_filter, h_filter, filter_width * filter_width, cudaMemcpyHostToDevice);

    rgb_image_t result;
    cudaMalloc(&result, height * width * CHANNELS);

    const dim3 block_size(16, 16, 1);
    const dim3 grid_size(width / block_size.x + 1, height / block_size.y + 1, 1);

    kernel_apply_filter<<<grid_size, block_size>>>(image, result, height, width, d_filter, filter_width);

    cudaMemcpy(image, result, height * width * CHANNELS, cudaMemcpyDeviceToDevice);
    cudaFree(result);
    cudaFree(d_filter);
    delete[] h_filter;
}