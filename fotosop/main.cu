#include <iostream>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main()
{
    cout << "Hello!" << endl;

    string input_file;
    string output_file;

    cout << "Image source: " << endl;
    cin >> input_file;
    cout << "Image dest: " << endl;
    cin >> output_file;

    uint8_t *d_rgb_image, *h_res_image, *d_res_image;
    int y_size, x_size;

    return 0;
}
