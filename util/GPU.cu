#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        //std::exit(EXIT_FAILURE);
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

typedef struct { signed char x, y, z, w, a, b, c, d; } char8;
__global__ void RG10toBW8(short* from, char* to,int width, int height){
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    int loc = x+width*y;
    if (x >= width || y >= height) return;
    to[loc] = (char)(from[loc]>>2);
}
__global__ void RG10toBW8_modulo4(short* from, char* to,int width, int height){
    int x = short_vec*(threadIdx.x+blockDim.x*blockIdx.x);
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    int loc = x+width*y;
    if (x >= width || y >= height) return;
    short4 vecInput_1 = ((const short4*)(&from[loc]))[0];
    char4 write{
        static_cast<signed char>((vecInput_1.x)>>2),
        static_cast<signed char>((vecInput_1.y)>>2),
        static_cast<signed char>((vecInput_1.z)>>2),
        static_cast<signed char>((vecInput_1.w)>>2)
    };
    ((char4*)(&to[loc]))[0] = write;
}
__global__ void RG10toBW8_modulo8(short* from, char* to,int width, int height){
    int x = char_vec*(threadIdx.x+blockDim.x*blockIdx.x);
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    int loc = x+width*y;
    if (x >= width || y >= height) return;
    short4 vecInput_1 = ((const short4*)(&from[loc]))[0];
    short4 vecInput_2 = ((const short4*)(&from[loc]))[1];
    char8 write{
        (static_cast<signed char>(vecInput_1.x>>2)),
        (static_cast<signed char>(vecInput_1.y>>2)),
        (static_cast<signed char>(vecInput_1.z>>2)),
        (static_cast<signed char>(vecInput_1.w>>2)),
        (static_cast<signed char>(vecInput_2.x>>2)),
        (static_cast<signed char>(vecInput_2.y>>2)),
        (static_cast<signed char>(vecInput_2.z>>2)),
        (static_cast<signed char>(vecInput_2.w>>2))
    };
    ((char8*)(&to[loc]))[0] = write;
}

#include "timer.h"
void process_GPU_RG10toBW8(char* data, char* result,int width,int height,int size){
    void* from = allocate_GPU(size);
    void* to   = allocate_GPU(height*width);
    memcopy_CPU_to_GPU(data,from,size);
    constexpr int threads = 32;
    constexpr dim3 thread(threads, threads);
    //Timer t;

    if (width % char_vec == 0){
        const dim3 grid(divup(width, threads*char_vec), divup(height, threads));
        RG10toBW8_modulo8<<<grid, thread,0>>>((short*)from,(char*)to,width,height);
    } else if (width % short_vec == 0){
        const dim3 grid(divup(width, threads*short_vec), divup(height, threads));
        RG10toBW8_modulo4<<<grid, thread,0>>>((short*)from,(char*)to,width,height);
    } else {
        const dim3 grid(divup(width, threads), divup(height, threads));
        RG10toBW8<<<grid, thread,0>>>((short*)from,(char*)to,width,height);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    //t.time_stamp();
    deallocate_GPU(from);
    memcopy_GPU_to_CPU(to,result,height*width);
    deallocate_GPU(to);
}
