#ifndef GPU_H
#define GPU_H

#include "image.h"
extern void process_GPU_RG10toBW8(char* from,char* result,int width,int height,int size); //#include "GPU.cu"

void RG10toBW_GPU(RG10 img,char* result){
    process_GPU_RG10toBW8(img.start,result,img.info.width,img.info.height,img.info.size());
}

#endif
