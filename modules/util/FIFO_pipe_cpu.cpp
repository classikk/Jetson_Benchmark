#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

void FIFO_pipe(int W, int H, std::vector<char> frame) {
    const int W = 320;
    const int H = 240;
    const char* pipeName = "/tmp/imagepipe";

    // Create FIFO if it doesn't exist
    mkfifo(pipeName, 0666);

    std::ofstream pipe(pipeName, std::ios::binary);
    if (!pipe) {
        std::cerr << "Failed to open pipe\n";
        exit(1);
    }

    pipe.write(reinterpret_cast<char*>(frame.data()), frame.size());
    pipe.flush();

}
