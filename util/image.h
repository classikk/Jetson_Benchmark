#ifndef IMAGE_H
#define IMAGE_H

using namespace std;


struct RG10  ;
struct BGR888;
struct RGB888;
struct IMG_info {
    int   width           ;
    int   height          ;
    int   pix_width       ;
    bool  GPU_pointer     ;
    int   data_width      ;
    int   data_height     ;
    int size() const {
        return pix_width*width*height;
    };

    IMG_info(int width, int height, int pix_width)
        :   width           (width          ), 
            height          (height         ),
            pix_width       (pix_width      ),
            GPU_pointer     (false          ),
            data_width      (width          ),
            data_height     (height         )
         {}
    IMG_info(int width, int height, int pix_width, bool GPU_pointer)
        :   width           (width          ), 
            height          (height         ),
            pix_width       (pix_width      ),
            GPU_pointer     (GPU_pointer    ),
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

    RGB888 toRGB(char* newstart) const;//char* newstart
    char* newCharArrToRGB() const;
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

RGB888 RG10::toRGB(char* newstart) const { 
    #pragma omp parallel for
    for(int y = 0; y < info.height; y++){
        for(int x = 0; x < info.width; x++){
            int pixel = y*info.width+x;
            char* write_to = newstart + RGB888::pix_width*pixel;
            write_to[0] = start[2*pixel];
            write_to[1] = start[2*pixel];
            write_to[2] = start[2*pixel];
        }
    }
    return RGB888(newstart,info); 
}
char* RG10::newCharArrToRGB() const {
    char* a = new char[RGB888::pix_width*info.width*info.height]; 
    return a; 
}
#endif