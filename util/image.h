#ifndef IMAGE_H
#define IMAGE_H

using namespace std;


struct RG10  ;
struct RGB888;
struct BW8;
struct IMG_info {
    int   width           ;
    int   height          ;
    int   pix_width       ;
    bool  Is_GPU_pointer  ;
    int   data_width      ;
    int   data_height     ;
    int size() const {
        return pix_width*width*height;
    };

    IMG_info(int width, int height, int pix_width)
        :   width           (width          ), 
            height          (height         ),
            pix_width       (pix_width      ),
            Is_GPU_pointer  (false          ),
            data_width      (width          ),
            data_height     (height         )
         {}
    IMG_info(int width, int height, int pix_width, bool Is_GPU_pointer)
        :   width           (width          ), 
            height          (height         ),
            pix_width       (pix_width      ),
            Is_GPU_pointer  (Is_GPU_pointer ),
            data_width      (width          ),
            data_height     (height         )
         {}
};


struct RG10 {
    char* start;
    IMG_info info;
    static constexpr int pix_width = 2;

    RG10(char* start, int width, int height)
        : start(start),info(width,height,pix_width) {}

    RG10(char* start, IMG_info new_info)
        : start(start),info(new_info) {info.pix_width = pix_width;}
    RG10()
        : start(start),info(info) {info.pix_width = pix_width;}

    RGB888 toRGB(char* newstart) const;
    BW8 toBW(char* newstart) const;
    char* newCharArrToRGB() const;
    char* newCharArrToBW() const;
};

struct RGB888 {
    char* start;
    IMG_info info;
    static constexpr int pix_width = 3;

    RGB888(char* start, int width, int height)
        : start(start),info(width,height,pix_width) {}

    RGB888(char* start, IMG_info new_info)
        : start(start),info(new_info) {info.pix_width = pix_width;}
};

struct BW8 {
    char* start;
    IMG_info info;
    static constexpr int pix_width = 1;

    BW8(char* start, int width, int height)
        : start(start),info(width,height,pix_width) {}

    BW8(char* start, IMG_info new_info)
        : start(start),info(new_info) {info.pix_width = pix_width;}
};

//#include <thread>
RGB888 RG10::toRGB(char* newstart) const { 
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //return RGB888(newstart,info); 
    #pragma omp parallel for
    for(int y = 0; y < info.height; y++){
        for(int x = 0; x < info.width; x++){
            int pixel = y*info.width+x;
            char* write_to = (char*)(newstart + RGB888::pix_width*pixel);
            write_to[0] = (char)(((short*)(start+2*pixel))[0]>>2);
            write_to[1] = (char)(((short*)(start+2*pixel))[0]>>2);
            write_to[2] = (char)(((short*)(start+2*pixel))[0]>>2);
        }
    }
    return RGB888(newstart,info); 
}

//extern void process_gpu(RG10 img,char* result); //#include "GPU.cu"

//#include <thread>
//#include "GPU.cu"
BW8 RG10::toBW(char* newstart) const { 
    //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //return RGB888(newstart,info); 
    #pragma omp parallel for
    for(int y = 0; y < info.height; y++){
        for(int x = 0; x < info.width; x++){
            int pixel = y*info.width+x;
            char* write_to = (char*)(newstart + BW8::pix_width*pixel);
            write_to[0] = (char)(((short*)(start+2*pixel))[0]>>2);
        }
    }
    //process_gpu(this[0],newstart);
    return BW8(newstart,info); 
}
char* RG10::newCharArrToRGB() const {
    char* a = new char[RGB888::pix_width*info.width*info.height]; 
    return a; 
}
char* RG10::newCharArrToBW() const {
    char* a = new char[BW8::pix_width*info.width*info.height]; 
    return a; 
}
#endif