#ifndef DATATYPES_H
#define DATATYPES_H

#include <cstdint>

#define CHANNELS 3

typedef const uint8_t *crgb_image_t;
typedef uint8_t *rgb_image_t;

typedef const uint8_t *cgray_image_t;
typedef uint8_t *gray_image_t;

enum ConversionType {
    GRAYSCALE = 0
};

enum AdditiveType {
    CONTRAST = 0
};

typedef struct rgb {
    float r, g, b;
} rgb_t;

typedef struct hsl {
    float h, s, l;
} hsl_t;

#endif