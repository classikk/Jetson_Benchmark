#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

void* allocate_GPU(int n_bytes){
    void* dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, n_bytes));
    return dataGPU;
}

void deallocate_GPU(void* mem){
    CHECK(cudaFree(mem));
}

void memcopy_GPU_to_CPU(void* from, void* to, int n_bytes){
    CHECK(cudaMemcpy(to, from, n_bytes, cudaMemcpyDeviceToHost));
}

void memcopy_CPU_to_GPU(void* from, void* to, int n_bytes){
    CHECK(cudaMemcpy(to, from, n_bytes, cudaMemcpyHostToDevice));
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}


constexpr int short_vec = 4;
constexpr int char_vec = 8;
typedef struct { char x, y, z, w, a, b, c, d; } char8;
__global__ void RG10toBW8(short* from, char* to,int width, int height){
    int x = 4*(threadIdx.x+blockDim.x*blockIdx.x);
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    if (x >= width || y >= height) return;
    int loc = x+width*y;
    //to[loc] = (char)(from[loc]>>2);
    short4 vecInput_1 = ((const short4*)(&from[loc]))[0];
    char4 write{
        (unsigned char)((vecInput_1.x)>>2),
        (unsigned char)((vecInput_1.y)>>2),
        (unsigned char)((vecInput_1.z)>>2),
        (unsigned char)((vecInput_1.w)>>2)
    };
    ((char4*)(&to[loc]))[0] = write;
}
__global__ void RG10toBW8_v2_modulo4(short* from, char* to,int width, int height){
    int x = 4*(threadIdx.x+blockDim.x*blockIdx.x);
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    if (y >= height) return;
    int loc = x+width*y;
    //to[loc] = (char)(from[loc]>>2);
    short4 vecInput_1 = ((const short4*)(&from[loc]))[0];
    char4 write{
        (unsigned char)((vecInput_1.x)>>2),
        (unsigned char)((vecInput_1.y)>>2),
        (unsigned char)((vecInput_1.z)>>2),
        (unsigned char)((vecInput_1.w)>>2)
    };
    ((char4*)(&to[loc]))[0] = write;
}
__global__ void RG10toBW8_v2_modulo8_thread_only(short* from, char* to,int width, int height){
    int n = short_vec*(threadIdx.x);
    if (n >= width*height){
        return;
    }
    for (int i = n;i < width*height; i += short_vec*blockDim.x){
        //to[i] = (char)(from[i]>>2);
        char4* write_to = reinterpret_cast<char4*>(to+i);
        short4 vecInput_1 = (reinterpret_cast<const short4*>(from+i))[0];
        //short4 vecInput_2 = (reinterpret_cast<const short4*>(from+i))[1];
        //char8 write{
        //    (char)((vecInput_1.x)>>2),
        //    (char)((vecInput_1.y)>>2),
        //    (char)((vecInput_1.z)>>2),
        //    (char)((vecInput_1.w)>>2),
        //    (char)((vecInput_2.x)>>2),
        //    (char)((vecInput_2.y)>>2),
        //    (char)((vecInput_2.z)>>2),
        //    (char)((vecInput_2.w)>>2)
        //};
        char4 write{
            (char)((vecInput_1.x)>>2),
            (char)((vecInput_1.y)>>2),
            (char)((vecInput_1.z)>>2),
            (char)((vecInput_1.w)>>2)
        };
        //short4 write{
        //    ((vecInput_1.x)>>2)+(((vecInput_1.y)>>2)<<8),
        //    ((vecInput_1.z)>>2)+(((vecInput_1.w)>>2)<<8),
        //    ((vecInput_2.x)>>2)+(((vecInput_2.y)>>2)<<8),
        //    ((vecInput_2.z)>>2)+(((vecInput_2.w)>>2)<<8)
        //};
        //write_to[0] = (char)((vecInput_1.x)>>2);
        //write_to[1] = (char)((vecInput_1.y)>>2);
        //write_to[2] = (char)((vecInput_1.z)>>2);
        //write_to[3] = (char)((vecInput_1.w)>>2);
        //write_to[4] = (char)((vecInput_2.x)>>2);
        //write_to[5] = (char)((vecInput_2.y)>>2);
        //write_to[6] = (char)((vecInput_2.z)>>2);
        //write_to[7] = (char)((vecInput_2.w)>>2);
        write_to[0] = write;
    }
}
#include "timer.h"
void process_GPU_RG10toBW8(char* data, char* result,int width,int height,int size){
    void* from = allocate_GPU(size);
    void* to   = allocate_GPU(height*width);
    memcopy_CPU_to_GPU(data,from,size);
    Timer t;
    if (true || width*height % 8 != 0){
        constexpr int threads = 32;
        constexpr dim3 thread(threads, threads);
        const dim3 grid(divup(width, threads*4), divup(height, threads));
        RG10toBW8<<<grid, thread,0>>>((short*)from,(char*)to,width,height);
        CHECK(cudaGetLastError());
    } else {
        RG10toBW8_v2_modulo8_thread_only<<<1, 1024, 0>>>((short*)from,(char*)to,width,height);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    t.time_stamp();
    deallocate_GPU(from);
    memcopy_GPU_to_CPU(to,result,height*width);
    deallocate_GPU(to);
}

