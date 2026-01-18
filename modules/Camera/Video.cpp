#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <stdexcept>

using RGBFrame = std::vector<char>;

struct VideoData {
    int width;
    int height;
    std::vector<RGBFrame> frames;
};

VideoData extractFramesAsRGB(const std::string& filename) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video file: " + filename);
    }

    VideoData result;

    // Read resolution from metadata
    result.width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    result.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::Mat frame, rgb;

    while (true) {
        if (!cap.read(frame)) {
            break;  // No more frames
        }

        // Convert BGR → RGB
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        // Flatten into a vector<uint8_t>
        RGBFrame buffer;
        buffer.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());

        result.frames.push_back(std::move(buffer));
    }

    return result;
}

#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <functional>
#include <tuple>
#include "../util/pipe.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <pipeName> <videoName>\n";
        return 1;
    }
    const char* pipeName = argv[1];
    const char* videoName = argv[2];
    
    auto video = extractFramesAsRGB(videoName);
    auto index = 0;
    PointIntFunc getVideo = [&video,&index]() -> std::tuple<char*, int> {
        if (video.frames.size() == index) index = 0;
        return { video.frames[index].data(), video.width*video.height*3 };
        index++;
    };

    createPipe(pipeName,getVideo);
    
    return 0;
}
