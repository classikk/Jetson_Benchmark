#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

int main() {
    const int W = 320;
    const int H = 240;
    const char* pipeName = "/tmp/imagepipe";

    // Create FIFO if it doesn't exist
    mkfifo(pipeName, 0666);

    std::ofstream pipe(pipeName, std::ios::binary);
    if (!pipe) {
        std::cerr << "Failed to open pipe\n";
        return 1;
    }

    std::vector<unsigned char> frame(W * H);

    int t = 0;
    while (true) {
        // Generate a simple moving pattern
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                frame[y * W + x] = ((x + t) % 256); // grayscale gradient
            }
        }

        pipe.write(reinterpret_cast<char*>(frame.data()), frame.size());
        pipe.flush();

        usleep(30000); // ~33 FPS
        t++;
    }

    return 0;
}
