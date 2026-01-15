#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    const int W = 320;
    const int H = 240;
    const char* pipeName = "/tmp/imagepipe";

    std::ifstream pipe(pipeName, std::ios::binary);
    if (!pipe) {
        std::cerr << "Failed to open pipe\n";
        return 1;
    }

    std::vector<unsigned char> buffer(W * H);

    cv::namedWindow("Viewer", cv::WINDOW_AUTOSIZE);

    while (true) {
        pipe.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        if (!pipe) break;

        cv::Mat img(H, W, CV_8UC1, buffer.data());

        cv::imshow("Viewer", img);

        if (cv::waitKey(1) == 27) break; // ESC to exit
    }

    return 0;
}
