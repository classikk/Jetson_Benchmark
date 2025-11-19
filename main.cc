#include <iostream>
#include <thread>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"

using namespace std;

int main() {
    Timer t;

    int width = 640;   // your image width
    int height = 480;  // your image height

    // Suppose this is your raw RGB888 buffer in RAM
    char* buffer = new char[width * height * 3];

    t.time("");
    RGB888 img = RGB888{buffer,width,height};
    for (int r = 0; r < 255; r++){
        Timer t;
        for (int i = 0; i < width * height; i++) {
            buffer[3*i + 0] = r; 
            buffer[3*i + 1] = 0;   
            buffer[3*i + 2] = 0;   
        }
        t.time("before");
        display(img);
        t.time("after");
    }
    t.time("");
    this_thread::sleep_for(std::chrono::seconds(2));

    delete [] buffer;

    return 0;
}