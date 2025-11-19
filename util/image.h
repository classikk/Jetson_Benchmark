#ifndef IMAGE_H
#define IMAGE_H

using namespace std;

struct RG10 {
    const char* start;
    const int width;
    const int height;
    const int pix_width = 2;
    const int size = pix_width*width*height;

    RG10(char* start,int width,int height)
        : start(start),width(width),height(height),size(pix_width*width*height) {}
};

struct RGB888 {
    const char* start;
    const int width;
    const int height;
    const int pix_width = 3;
    const int size = pix_width*width*height;

    RGB888(char* start,int width,int height)
        : start(start),width(width),height(height),size(pix_width*width*height) {}
};

#endif