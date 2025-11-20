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
        return -1;
    };
    int total = 0;
    char* rgb = new char[3*1280*720];
    while (t.seconds() < 10.0){
        total += 1;
        RG10 raw_frame = video_stream.get_frame();
        if (rgb == NULL){
            rgb = raw_frame.newCharArrToRGB();
        }
        //t.time_stamp("1");
        RGB888 frame = raw_frame.toRGB(rgb);
        //t.time_stamp("2");
        video_stream.record_new_image();
        //t.time_stamp("3");
        display(frame);
        //t.time_stamp("4");   
        //cout << total/t.seconds() << "fps" << endl;
    }
    cout << total/t.seconds() << "fps" << endl;
    delete [] rgb;
    
    video_stream.cleanUp();

    return 0;
}
