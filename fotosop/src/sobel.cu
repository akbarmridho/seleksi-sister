#include "sobel.cuh"
#include "converter.cuh"

__global__ void sobel_gpu(const uint8_t *source, uint8_t *result, const int width, const int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float dx, dy;

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        dx = (-1 * source[CHANNELS * ((y - 1) * width + (x - 1))]) +
             (-2 * source[CHANNELS * (y * width + (x - 1))]) +
             (-1 * source[CHANNELS * ((y + 1) * width + (x - 1))]) +
             (source[CHANNELS * ((y - 1) * width + (x + 1))]) +
             (2 * source[CHANNELS * (y * width + (x + 1))]) +
             (source[CHANNELS * ((y + 1) * width + (x + 1))]);
        dy = (source[CHANNELS * ((y - 1) * width + (x - 1))]) +
             (2 * source[CHANNELS * ((y - 1) * width + x)]) +
             (source[CHANNELS * ((y - 1) * width + (x + 1))]) +
             (-1 * source[CHANNELS * ((y + 1) * width + (x - 1))]) +
             (-2 * source[CHANNELS * ((y + 1) * width + x)]) +
             (-1 * source[CHANNELS * ((y + 1) * width + (x + 1))]);

        uint8_t value = max(0, min(255, (uint8_t) floor(sqrt((dx * dx) + (dy * dy)))));

        result[(y * width + x) * CHANNELS + 0] = value;
        result[(y * width + x) * CHANNELS + 1] = value;
        result[(y * width + x) * CHANNELS + 2] = value;
    }
}


void sobel_edge(rgb_image_t image, int height, int width) {
    rgb_image_t result;
    cudaMalloc(&result, height * width * CHANNELS);

    to_grey(image, height, width);

    const dim3 block_size(16, 16, 1);
    const dim3 grid_size(width / block_size.x + 1, height / block_size.y + 1, 1);

    sobel_gpu<<<grid_size, block_size>>>(image, result, width, height);

    cudaMemcpy(image, result, height * width * CHANNELS, cudaMemcpyDeviceToDevice);
    cudaFree(result);
}