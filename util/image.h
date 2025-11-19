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

    RGB888 toRGB() const;//char* newstart
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

RGB888 RG10::toRGB() const { //char* newstart
    char* result = new char[3*width*height];//RGB888((char*)this,-1,1).pix_width
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            int pixel = y*width+x;
            char* write_to = result + 3*pixel;
            write_to[0] = (start[2*pixel]);
            write_to[1] = (start[2*pixel]);
            write_to[2] = (start[2*pixel]);
        }
    }
    return RGB888(result,width,height); //newstart
}

#endif