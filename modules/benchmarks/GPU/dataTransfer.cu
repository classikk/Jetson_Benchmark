#include <cuda_runtime.h>
#include "../../util/GPU.h"
#include "../../kernels/kernel.h"
#include "../../util/timer.h"

#include <iostream>
#include <signal.h>
volatile sig_atomic_t stop = 0;
void handle_sigint(int sig) {
    stop = 1;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handle_sigint);
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <pixel bytes>\n";
        return 1;
    }

    int w = std::atoi(argv[1]);
    int h = std::atoi(argv[2]);
    int p = std::atoi(argv[3]);
    int size = w*h*p;

    char* data = new char[size];
    for (int i = 0; i < size; i++) (data)[i] = i;
    void* gpu = allocate_GPU(size);
    BenchMark t;
    int n = 0;
    while (!stop){
        n++;
        memcopy_CPU_to_GPU(data,gpu,size);
        synchronize();
        t.step_Completed("memcopy_CPU_to_GPU");

        //int dummyMatrix = (w>h? h : w )/3;
        //dummyMatrix = dummyMatrix > 1 ? 1 : dummyMatrix;
        //matMul((float*)gpu,(float*)gpu,(float*)gpu,dummyMatrix,dummyMatrix,dummyMatrix); 
        //afterKernelCallCheckErrors();
        //synchronize();
        //t.step_Completed("random operation to ensure it does not get optimized away.");

        memcopy_GPU_to_CPU(gpu,data,size);
        synchronize();
        t.cycle_Completed("memcopy_GPU_to_CPU");
    }
    deallocate_GPU(gpu);
    delete [] data;
    std::cout << "\nTransfering " << size*t.fps()/1000/1000 <<"MB/s CPU to GPU and back" << endl;
    return 0;
}


        //cudaEvent_t start,stop;
        //cudaEventCreate(&start);
        //cudaEventCreate(&stop);
        //cudaEventRecord(start);

        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //float ms = 0.0f;
        //cudaEventElapsedTime(&ms,start,stop);
        //time += ms;
    //float time = 0.0f;
    //std::cout << time/n << endl;

        //this is just to check things don't get optimized away.
        //dummyTestKernel(gpu,size);
        //int dummyMatrix = (w>h? h : w )/3;
        //dummyMatrix = dummyMatrix > 1 ? 1 : dummyMatrix;
        //matMul((float*)gpu,(float*)gpu,(float*)gpu,1,1,1); 
        //afterKernelCallCheckErrors();
        //synchronize();
        //t.step_Completed("random operation to ensure it does not get optimized away.");

    //std::cout << "thingy to check it does not optimize it away. : " << data[0] << sqrtf64(sqrtf64(size)/2)<< endl;