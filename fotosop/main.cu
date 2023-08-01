#include <iostream>
#include "src/loader.h"
#include "src/converter.cuh"
#include "src/blur.cuh"

#define CVUI_IMPLEMENTATION
#define WINDOW_NAME "Fotosop"

#include "lib/cvui.h"
#include "lib/osdialog.h"

/**
 * Change contrast value range to [-255, 255] or saturation to [-1,1] if you like
 */
#define CONTRAST_MAX 50
#define CONTRAST_MIN (-50)
#define SATURATION_MAX 1.0
#define SATURATION_MIN (-1.0)

using std::cin;
using std::cout;
using std::endl;
using std::string;

inline bool ends_with(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

int main() {
    cv::Mat main_frame = cv::Mat(540, 1250, CV_8UC3);

    cv::Mat loaded_image;
    string loaded_path;

    cv::Mat result;
    cv::Mat result_resized;
    bool has_resized_result = false;

    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);

    int applied_contrast = 0;
    float applied_saturation = 0.0;
    FILTER_TYPE applied_filter = NONE;

    int contrast_value = 0.0;
    float saturation_value = 0.0;

    bool grayscale_filter_checked = false;
    bool edge_filter_checked = false;
    bool blur_filter_checked = false;
    FILTER_TYPE current_filter = NONE;

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

        cvui::printf(main_frame, 970, 100, 0.4, 0xffffff, "Contrast");
        cvui::trackbar(main_frame, 970, 125, 250, &contrast_value, CONTRAST_MIN, CONTRAST_MAX, 1, "%.1Lf",
                       cvui::TRACKBAR_DISCRETE);

        cvui::printf(main_frame, 970, 180, 0.4, 0xffffff, "Saturation");
        cvui::trackbar(main_frame, 970, 205, 250, &saturation_value, (float) SATURATION_MIN, (float) SATURATION_MAX);

        cvui::checkbox(main_frame, 970, 290, "Edge", &edge_filter_checked);
        cvui::checkbox(main_frame, 970, 320, "Blur", &blur_filter_checked);
        cvui::checkbox(main_frame, 970, 260, "Grayscale", &grayscale_filter_checked);

        cvui::printf(main_frame, 970, 350, 0.4, 0xffffff, "Pilihan paling atas akan diprioritaskan.");

        if (edge_filter_checked) {
            current_filter = EDGE;
        } else if (blur_filter_checked) {
            current_filter = BLUR;
        } else if (grayscale_filter_checked) {
            current_filter = GRAYSCALE;
        } else {
            current_filter = NONE;
        }

        // TODO or applied filter is different
        if (!loaded_image.empty() && (
                result.empty() ||
                applied_contrast != contrast_value ||
                applied_saturation != saturation_value ||
                current_filter != applied_filter
        )) {
            rgb_image_t rgb_image;
            int height, width;
            auto total_pixels = load_image(&rgb_image, loaded_image, &height, &width);

            // host - cpu
            // device - gpu
            // rgb image in cuda memory
            rgb_image_t d_rgb_image;
            cudaMalloc(&d_rgb_image, sizeof(uint8_t) * total_pixels * CHANNELS);
            cudaMemcpy(d_rgb_image, rgb_image, sizeof(uint8_t) * total_pixels * CHANNELS, cudaMemcpyHostToDevice);

            add_contrast(d_rgb_image, height, width, contrast_value);

            add_saturation(d_rgb_image, height, width, saturation_value);

            if (current_filter == GRAYSCALE) {
                to_grey(d_rgb_image, height, width);
            } else if (current_filter == BLUR) {
                gaussian_blur(d_rgb_image, height, width);
            }

            cudaMemcpy(rgb_image, d_rgb_image, sizeof(uint8_t) * total_pixels * CHANNELS, cudaMemcpyDeviceToHost);

            result = cv::Mat(height, width, CV_8UC3);
            std::memcpy(result.data, rgb_image, total_pixels * sizeof(uint8_t) * CHANNELS);
            free(rgb_image);
            cudaFree(d_rgb_image);

            has_resized_result = false;
            applied_saturation = saturation_value;
            applied_contrast = contrast_value;
            applied_filter = current_filter;
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

    return 0;
}
