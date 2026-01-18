#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using RGBFrame = std::vector<uint8_t>;  
// Stores pixels as [R,G,B, R,G,B, ...] row-major

std::vector<RGBFrame> extractFramesAsRGB(const std::string& filename) {
    std::vector<RGBFrame> frames;

    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video file: " + filename);
    }

    cv::Mat frame, rgb;
    while (true) {
        if (!cap.read(frame)) {
            break;  // No more frames
        }

        // Convert BGR (OpenCV default) → RGB
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        // Flatten into a vector<uint8_t>
        RGBFrame buffer;
        buffer.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());

        frames.push_back(std::move(buffer));
    }

    return frames;
}
