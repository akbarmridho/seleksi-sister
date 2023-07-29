# Fotosop

## Development Environment

This project was developed and compiled the following environment:

- WSL 2 Ubuntu 22.04
- CUDA 12.2
- AMD Ryzen 7 4800H, Nvidia RTX 3050
- Libgtk3. Install with `sudo apt-get install libgtk2.0-dev libgtk-3-dev`
- OpenCV 4.8. Install with `sudo apt install libopencv-dev` or compile from release

## Prerequisites

- CUDA Capable hardware
- Ubuntu
- OpenCV 4.8
- libgtk3
- Cmake 3.24
- C and C++ compiler that support C17 and C++ 17

## How to Run

- Run the compiled binary see if it works.
- If not, compile from the project

## How to Compile

- From current folder, create a new folder named build with `mkdir build`
- Run cmake `cmake -DCMAKE_BUILD_TYPE=Release . -B build`
- Inside build folder, run make `cd build && make`
- Run the compiled binary

## User Interface

- Open file button, to select an image for filtering
- Save file button, to save processed image
- Contrast slider, to modify contrast value
- Saturation slider, to modify image saturation
- Grayscale, edge, and blur checkbox, to apply corresponding filter to the image.

## Algorithm

Every algorithm below was implemented in parallel with CUDA.

- Grayscale -> set RGB value to its average value for that pixels.
- Contrast -> set RGB value based on `max(0, min(255, c * (v - 128) + 128))` formula where c is the contrast value.
- Saturation -> convert RGB to HSL, then multiply saturation value by `1 + s` where s is the saturation additional
  value, then convert it back to RGB.

## Specification and Bonus Checkbox

### Base

- [x] Grayscale
- [x] Contrast
- [x] Saturation

### Bonus

- [x] Parallelize base filtering algorithm with CUDA (3 points)
- [x] Real-time preview edit (2 points)
- [ ] Image from feed camera (2 points)
- [ ] Blur with GPU (8 points)
- [ ] Edge detection with GPU (15 points)

## Libraries

- File dialog, https://github.com/AndrewBelt/osdialog
- CVUI, https://github.com/Dovyski/cvui/