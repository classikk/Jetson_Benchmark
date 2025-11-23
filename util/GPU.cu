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


__global__ void RG10toBW8(short* from, char* to,int width, int height){
    int x = threadIdx.x+blockDim.x*blockIdx.x;
    int y = threadIdx.y+blockDim.y*blockIdx.y;
    if (x >= width || y >= height){
        return;
    }
    int loc = x+width*y;
    to[loc] = (char)(from[loc]>>2);
}
constexpr int vec = 4;
__global__ void RG10toBW8_v2_modulo4_thread_only(short* from, char* to,int width, int height){
    int n = vec*(threadIdx.x);
    if (n >= width*height){
        return;
    }
    for (int i = n; i < width*height; i += blockDim.x*vec){
        short4 vecInput = (reinterpret_cast<const short4*>(from+i))[0];
        to[i+0] = (char)((vecInput.x)>>2);
        to[i+1] = (char)((vecInput.y)>>2);
        to[i+2] = (char)((vecInput.z)>>2);
        to[i+3] = (char)((vecInput.w)>>2);
    }
}
#include "timer.h"
void process_GPU_RG10toBW8(char* data, char* result,int width,int height,int size){
    void* from = allocate_GPU(size);
    void* to   = allocate_GPU(height*width);
    memcopy_CPU_to_GPU(data,from,size);
    Timer t;
    if (width*height % 4 != 0){
        constexpr int threads = 32;
        constexpr dim3 thread(threads, threads);
        const dim3 grid(divup(width, threads), divup(height, threads));
        RG10toBW8<<<grid, thread,0>>>((short*)from,(char*)to,width,height);
        CHECK(cudaGetLastError());
    } else {
        RG10toBW8_v2_modulo4_thread_only<<<1, 1024, 0>>>((short*)from,(char*)to,width,height);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    t.time_stamp();
    deallocate_GPU(from);
    memcopy_GPU_to_CPU(to,result,height*width);
    deallocate_GPU(to);
}

