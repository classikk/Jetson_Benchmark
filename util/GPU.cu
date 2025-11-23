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

#include "image.h"
constexpr int threads = 32;

__global__ void RG10toBW8(short* from, char* to,int width, int height){
    int x = threadIdx.x+threads*blockIdx.x;
    int y = threadIdx.y+threads*blockIdx.y;
    if (x >= width || y >= height){
        return;
    }
    int loc = x+width*y;
    to[loc] = (char)(from[loc]>>2);
}

void process_gpu(RG10 img,char* result){
    void* from = allocate_GPU(img.info.size());
    void* to   = allocate_GPU(img.info.height*img.info.width);
    memcopy_CPU_to_GPU(img.start,from,img.info.size());
    constexpr dim3 thread(threads, threads);
    const dim3 grid(divup(img.info.width, threads), divup(img.info.height, threads));
    //if (!img.info.Is_GPU_pointer)
    RG10toBW8<<<grid, thread,0>>>((short*)from,(char*)to,img.info.width,img.info.height);
    deallocate_GPU(from);
    memcopy_GPU_to_CPU(to,result,img.info.height*img.info.width);
    deallocate_GPU(to);
}
