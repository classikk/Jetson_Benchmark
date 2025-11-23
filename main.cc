#include <iostream>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"
#include "util/camera.h"
//Just used to shush the false positive leaks.
#include "util/GPU.h"
static inline void GPU_cleanUp(){
    cleanUp();
}


using namespace std;
int main() {
    Timer t;
    Streamer video_stream(false);
    if (!video_stream.init_success){
        cout << "failed initialisation" << endl;
        return -1;
    };
    
    char* rgb = video_stream.get_frame().newCharArrToRGB();
    while (t.seconds() < 300.0){
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
    }
    GPU_cleanUp();
    t.show_benchmark();
    t.fps();
    delete [] rgb;
    //GPU_cleanUp();
    video_stream.cleanUp();
    return 0;
}