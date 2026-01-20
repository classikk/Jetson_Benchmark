#include <cuda_runtime.h>
#include "../../util/GPU.cu"
#include "../../util/timer.h"

#include <iostream>
#include <signal.h>
volatile sig_atomic_t stop = 0;
void handle_sigint(int sig) {
    stop = 1;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handle_sigint);
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <bytes> <times n>\n";
        return 1;
    }

    int size = std::atoi(argv[1]);
    int n_times = std::atoi(argv[2]);

    int* data = new int[size+n_times];
    for (int i = 0; i < size+n_times; i++) (data)[i] = i;
    void* gpu = allocate_GPU(size);
    BenchMark t;
    for (int i = 0; i < n_times && !stop; i++){
        memcopy_CPU_to_GPU(data,gpu,size);
        t.step_Completed("memcopy_CPU_to_GPU");
        memcopy_GPU_to_CPU(gpu,data+1,size);
        t.cycle_Completed("memcopy_GPU_to_CPU");
    }
    deallocate_GPU(gpu);
    delete [] data;
    return 0;
}
