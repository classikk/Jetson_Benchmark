#include <iostream>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"
#include "util/camera.h"


#include <vector>

void fillTestImage(std::vector<char>& frame, int w, int h)
{
    frame.resize(w * h);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            bool white = ((x / 8) % 2) ^ ((y / 8) % 2);
            frame[y * w + x] = white ? 255 : 0;
        }
    }
}


#include <thread>
using namespace std;
int main() {
    std::vector<char> frame;
    fillTestImage(frame, 800,600);
    BW8 img = BW8(frame.data(),800,600);
    display(img);
    sleep(10);
    display(img);
    sleep(10);
    display(img);
    sleep(10);
    //Streamer video_stream(false);
    //if (!video_stream.init_success){
    //    cout << "failed initialisation" << endl;
    //    return -1;
    //};
    //char* rgb = video_stream.get_frame().newCharArrToRGB();
////
    //Timer t;
    //while (t.seconds() < 10.0){
    //    cout << "cycle" << endl;
    //    t.benchmark(0);
    //    RG10 raw_frame = video_stream.get_frame();
    //    t.benchmark(1);
    //    BW8 frame    = raw_frame.toBW(rgb);
    //    t.benchmark(2);
    //    display(frame);
    //    t.benchmark(3);
    //    t.total_frames += 1;
    //    t.show_benchmark();
    //    t.fps();
    //}
//
    //t.show_benchmark();
    //t.fps();
    //delete [] rgb;
    //video_stream.cleanUp();
    return 0;
}