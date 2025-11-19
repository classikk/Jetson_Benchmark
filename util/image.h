#ifndef IMAGE_H
#define IMAGE_H

using namespace std;


struct RG10  ;
struct BGR888;
struct RGB888;

struct RG10 {
    char* start;
    const int width;
    const int height;
    const int pix_width = 2;
    const int size = pix_width*width*height;

    RG10(char* start,int width,int height)
        : start(start),width(width),height(height),size(pix_width*width*height) {}

    RGB888 toRGB(char* newstart) const;
};

struct RGB888 {
    char* start;
    const int width;
    const int height;
    const int pix_width = 3;
    const int size = pix_width*width*height;

    RGB888(char* start,int width,int height)
        : start(start),width(width),height(height),size(pix_width*width*height) {}

};

RGB888 RG10::toRGB(char* newstart) const {
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
        }
    }
    return RGB888(newstart,width,height);
}

#endif