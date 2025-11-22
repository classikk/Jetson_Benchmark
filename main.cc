#include <iostream>
#include <thread>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"
#include "util/camera.h"

using namespace std;

int main() {
    Timer t;
    Streamer video_stream(false);
    if (!video_stream.init_success){
        cout << "failed initialisation" << endl;
        return -1;
    };
    int total = 0;
    char* rgb = new char[3*1280*720];
    while (t.seconds() < 5.0){
        total += 1;
        t.benchmark(0);
        RG10 raw_frame = video_stream.get_frame();
        video_stream.record_new_image();
        if (rgb == NULL){
            rgb = raw_frame.newCharArrToRGB();
        }
        t.benchmark(1);
        //t.time_stamp("1");
        RGB888 frame = raw_frame.toRGB(rgb);
        //t.time_stamp("2");
        t.benchmark(2);
        //display(raw_frame);
        //t.time_stamp("3");
        t.benchmark(3);
        display(frame);
        //t.time_stamp("4");   
        t.benchmark(4);
        //cout << total/t.seconds() << "fps" << endl;
    }
    t.show_benchmark();
    cout << total/t.seconds() << "fps" << endl;
    delete [] rgb;
    
    video_stream.cleanUp();

    return 0;
}
