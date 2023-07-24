#include <iostream>
#include <filesystem>
#include "src/loader.cuh"
#include "src/converter.cuh"

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
    cout << "Hello!" << endl;
    cout << "current path" << std::filesystem::current_path() << endl;

    string input_file;
    string output_file;

    cout << "Image source: " << endl;
    cin >> input_file;
    cout << "Image dest: " << endl;
    cin >> output_file;

    // host - cpu
    // device - gpu

    rgb_image_t h_rgb, d_rgb;
    gray_image_t h_res, d_res;
    int height, width;

    // load h_rgb
    auto total_pixels = load_image(&h_rgb, input_file, &height, &width);
    // copy h_rgb to d_rgb
    cudaMalloc(&d_rgb, sizeof(uint8_t) * total_pixels * CHANNELS);
    cudaMemcpy(d_rgb, h_rgb, sizeof(uint8_t) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);

    // allocate d_res
    cudaMalloc(&d_res, sizeof(uint8_t) * total_pixels);
    cudaMemset(d_res, 0, sizeof(uint8_t) * total_pixels);

    // convert
    to_grey(d_rgb, d_res, height, width);

    // copy result device to host
    h_res = (uint8_t *) malloc(sizeof(uint8_t) * total_pixels);
    cudaMemcpy(h_res, d_res, sizeof(uint8_t) * total_pixels, cudaMemcpyDeviceToHost);

    cudaFree(d_res);
    cudaFree(d_rgb);

    save_image(output_file, h_res, height, width, CV_8UC1);

    return 0;
}
