#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <functional>
#include <tuple>
#include "../util/camera.h"
#include "../util/pipe.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pipeName>\n";
        return 1;
    }
    const char* pipeName = argv[1];

    Streamer video_stream(false);
    if (!video_stream.init_success){
        cout << "failed initialisation" << endl;
        return -1;
    };

    PointIntFunc getVideo = [&video_stream]() -> std::tuple<char*, int> {
        RG10 raw_frame = video_stream.get_frame();
        return { raw_frame.start, raw_frame.info.size() };
    };

    createPipe(pipeName,getVideo);
    
    video_stream.cleanUp();
    return 0;
}