#include <iostream>
#include <filesystem>
#include "src/loader.h"
#include "src/converter.cuh"

#define CVUI_IMPLEMENTATION
#define WINDOW_NAME "Fotosop"

#include "lib/cvui.h"
#include "lib/osdialog.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
    /**
     * height 540
width 960
container

padding tombol jadi width 1100
     */
    cv::Mat main_frame = cv::Mat(540, 1100, CV_8UC1);

    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);

    int count = 0;

    while (true) {
        // Fill background color
        main_frame = cv::Scalar(49, 52, 49);

        if (cvui::button(main_frame, 1000, 20, "Open File")) {
            auto filters = osdialog_filters_parse("Image:jpg,jpeg,png");
            auto path = osdialog_file(OSDIALOG_OPEN, nullptr, nullptr, filters);
            std::cout << path << std::endl;
            osdialog_filters_free(filters);
        }
//
//        // Show how many times the button has been clicked.
//        // Text at position (250, 90), sized 0.4, in red.
//        cvui::printf(main_frame, 250, 90, 0.4, 0xff0000, "Button click count: %d", count);

        // Update cvui internal stuff
        cvui::update();

        // Show everything on the screen
        cv::imshow(WINDOW_NAME, main_frame);

        // Check if ESC key was pressed
        if (cv::waitKey(20) == 27) {
            break;
        }
    }

//    cout << "Hello!" << endl;
//    cout << "current path" << std::filesystem::current_path() << endl;
//
//    string input_file;
//    string output_file;
//
//    cout << "Image source: " << endl;
//    cin >> input_file;
//    cout << "Image dest: " << endl;
//    cin >> output_file;
//
//    // host - cpu
//    // device - gpu
//
//    rgb_image_t h_rgb, d_rgb;
//    gray_image_t h_res, d_res;
//    int height, width;
//
//    // load h_rgb
//    auto total_pixels = load_image(&h_rgb, input_file, &height, &width);
//    // copy h_rgb to d_rgb
//    cudaMalloc(&d_rgb, sizeof(uint8_t) * total_pixels * CHANNELS);
//    cudaMemcpy(d_rgb, h_rgb, sizeof(uint8_t) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);
//
//    // allocate d_res
//    cudaMalloc(&d_res, sizeof(uint8_t) * total_pixels);
//    cudaMemset(d_res, 0, sizeof(uint8_t) * total_pixels);
//
//    // convert
//    to_grey(d_rgb, d_res, height, width);
//
//    // copy result device to host
//    h_res = (uint8_t *) malloc(sizeof(uint8_t) * total_pixels);
//    cudaMemcpy(h_res, d_res, sizeof(uint8_t) * total_pixels, cudaMemcpyDeviceToHost);
//
//    cudaFree(d_res);
//    cudaFree(d_rgb);
//
//    save_image(output_file, h_res, height, width, CV_8UC1);

    return 0;
}
