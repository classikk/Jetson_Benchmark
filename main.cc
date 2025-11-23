#include <iostream>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"
#include "util/camera.h"


using namespace std;
int main() {
    Streamer video_stream(false);
    if (!video_stream.init_success){
        cout << "failed initialisation" << endl;
        return -1;
    };
    
    char* rgb = video_stream.get_frame().newCharArrToRGB();
    Timer t;
    while (t.seconds() < 1.0){
        t.benchmark(0);
        RG10 raw_frame = video_stream.get_frame();
        t.benchmark(1);
        //RGB888 frame = raw_frame.toRGB(rgb);
        BW8 frame    = raw_frame.toBW(rgb);
        //display(raw_frame);
        t.benchmark(2);
        display(frame);
        t.benchmark(3);
        t.total_frames += 1;
        t.show_benchmark();
        t.fps();
    }
    t.show_benchmark();
    t.fps();
    delete [] rgb;
    video_stream.cleanUp();
    return 0;
}