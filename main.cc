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

    // Fill buffer with something (for demo, set all pixels to red)
    for (int i = 0; i < width * height; i++) {
        buffer[3*i + 0] = 255; 
        buffer[3*i + 1] = 0;   
        buffer[3*i + 2] = 0;   
    }
    RGB888 img = RGB888{buffer,width,height};
    display(img);
    cout << img.size << endl;
    t.time();

    delete [] buffer;

    return 0;
}