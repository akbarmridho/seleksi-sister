#include <iostream>
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

inline bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int main() {
    cv::Mat main_frame = cv::Mat(540, 1200, CV_8UC3);

    cv::Mat loaded_image;
    string loaded_path;

    cv::Mat result;
    cv::Mat result_resized;
    bool has_resized_result = false;

    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);


    while (true) {
        // Fill background color
        main_frame = cv::Scalar(49, 52, 49);

        if (cvui::button(main_frame, 970, 20, "Open File")) {
            auto filters = osdialog_filters_parse("Image:jpg,jpeg,png");
            char *image_path_chr = osdialog_file(OSDIALOG_OPEN, nullptr, nullptr, filters);
            osdialog_filters_free(filters);

            if (image_path_chr != nullptr) {
                string image_path(image_path_chr);

                if (!image_path.empty() && (loaded_path.empty() || image_path != loaded_path)) {
                    loaded_image = cv::imread(image_path);

                    // image read valid
                    if (!loaded_image.empty()) {
                        loaded_path = image_path;
                        result = cv::Mat();
                    }
                }
            }
        }

        if (cvui::button(main_frame, 970, 60, "Save File") && !result.empty()) {
            auto filters = osdialog_filters_parse("Image:jpg,jpeg,png");
            char *image_path_chr = osdialog_file(OSDIALOG_SAVE, nullptr, nullptr, filters);
            osdialog_filters_free(filters);

            if (image_path_chr != nullptr) {
                string image_path(image_path_chr);
                if (!ends_with(image_path, ".jpg") && !ends_with(image_path, ".jpeg") &&
                    !ends_with(image_path, ".png")) {
                    image_path += ".jpg";
                }

                if (!image_path.empty()) {
                    cv::imwrite(image_path, result);
                }
            }
        }

        // TODO or applied filter is different
        if (result.empty()) {
            rgb_image_t rgb_image;
            int height, width;
            auto total_pixels = load_image(&rgb_image, loaded_image, &height, &width);

            // apply filters

            result = cv::Mat(height, width, CV_8UC3);
            std::memcpy(result.data, rgb_image, total_pixels * sizeof(uint8_t) * CHANNELS);
            free(rgb_image);
            has_resized_result = false;
        }

        if (result.empty()) {
            cvui::printf(main_frame, 360, 250, 0.8, 0xffffff, "No image loaded");
        } else {
            if (!has_resized_result) {
                result_resized = resize(result, 960, 540);
                has_resized_result = true;
            }
            cvui::image(main_frame, 0, 0, result_resized);
        }

        // Update cvui internal stuff
        cvui::update();

        // Show everything on the screen
        cv::imshow(WINDOW_NAME, main_frame);

        if (cv::waitKey(20) == 27 || cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_AUTOSIZE) == -1) {
            break;
        }
    }

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
