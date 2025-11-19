#include <iostream>
#include <thread>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"
#include "util/camera.h"

using namespace std;

int main() {
    Timer t;
    Streamer video_stream;
    if (!video_stream.init()){
        cout << "failed initialisation" << endl;
    };
    while (t.seconds() < 10.0){
        t.time("");
        RG10 raw_frame = video_stream.get_frame();
        t.time("");
        RGB888 frame = raw_frame.toRGB();
        t.time("");
        display(frame);
        delete [] frame.start;
        
    }
    t.time("");

    video_stream.cleanUp();

    return 0;
}

/*

    int width = 640;   // your image width
    int height = 480;  // your image height

    this_thread::sleep_for(std::chrono::seconds(2));
    // Suppose this is your raw RGB888 buffer in RAM
    delete [] buffer;
    char* buffer = new char[width * height * 3];
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
*/