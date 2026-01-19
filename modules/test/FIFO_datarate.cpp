#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "../util/pipe.cpp"
#include "../util/timer.h"

#include <signal.h>
volatile sig_atomic_t stop = 0;
void handle_sigint(int sig) {
    stop = 1;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handle_sigint);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <pixel_width> <pipeName>\n";
        return 1;
    }

    int W = std::atoi(argv[1]);
    int H = std::atoi(argv[2]);
    int pixel_width = std::atoi(argv[3]);
    const char* pipeName = argv[4];

    std::cout << "Width: " << W << "\n";
    std::cout << "Height: " << H << "\n";
    std::cout << "pixel_width: " << H << "\n";
    std::cout << "Pipe: " << pipeName << "\n";

    std::ifstream pipe = makePipe(pipeName);
    
    int size = W * H * pixel_width;

    cv::namedWindow("Viewer", cv::WINDOW_AUTOSIZE);
    BenchMark t;
    while (!stop) {
        auto buffer = usePipe(pipe,size);
        t.cycle_Completed();
    }

    return 0;
}
