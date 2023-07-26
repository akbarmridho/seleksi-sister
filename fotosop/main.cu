#include <iostream>
#include <filesystem>
#include "src/loader.h"
#include "src/converter.cuh"

#define CVUI_IMPLEMENTATION
#define WINDOW_NAME "Fotosop"
#define WINDOR_IMAGE "Image"

#include "lib/cvui.h"
#include "lib/osdialog.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;

cv::Mat resize(const cv::Mat &source, int width, int height) {
    int original_width = source.cols;
    int original_height = source.rows;

    float ratio = MIN(height / (1.0 * original_height), width / (1.0 * original_width));

    cv::Mat result;
    cv::resize(source, result, cv::Size(int(original_width * ratio), int(original_height * ratio)), cv::INTER_CUBIC);
    return result;
}

int main() {
    /**
     * height 540
width 960
container

padding tombol jadi width 1100
     */
    cv::Mat main_frame = cv::Mat(540, 1100, CV_8UC3);

    cv::Mat loaded_image;
    string loaded_path;

    cv::Mat result;
    cv::Mat result_resized;
    bool result_drawn = false;

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


        // TODO or applied filter is different
        if (result.empty()) {
            rgb_image_t rgb_image;
            int height, width;
            auto total_pixels = load_image(&rgb_image, loaded_image, &height, &width);

            // apply filters

            result = cv::Mat(height, width, CV_8UC3);
            std::memcpy(result.data, rgb_image, total_pixels * sizeof(uint8_t) * CHANNELS);
            free(rgb_image);
            result_drawn = false;
        }

        if (result.empty()) {
            cvui::printf(main_frame, 360, 250, 0.8, 0xffffff, "No image loaded");
        } else {
            if (!result_drawn) {
                result_resized = resize(result, 960, 540);
                result_drawn = true;
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
