#ifndef DATATYPES_H
#define DATATYPES_H

#include <cstdint>

#define CHANNELS 3

typedef uint8_t *rgb_image_t;

typedef struct rgb {
    float r, g, b;
} rgb_t;

typedef struct hsl {
    float h, s, l;
} hsl_t;

#endif