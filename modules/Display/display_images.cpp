#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "../util/pipe.cpp"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <isRGB> <pipeName>\n";
        return 1;
    }

    int W = std::atoi(argv[1]);
    int H = std::atoi(argv[2]);
    bool RGB = std::atoi(argv[3]) == 1;
    const char* pipeName = argv[4];

    std::cout << "Width: " << W << "\n";
    std::cout << "Height: " << H << "\n";
    std::cout << "Pipe: " << pipeName << "\n";

    std::ifstream pipe = makePipe(pipeName);
    
    int size = W * H * (RGB ? 3 : 2);

    cv::namedWindow("Viewer", cv::WINDOW_AUTOSIZE);

    while (true) {
        auto buffer = usePipe(pipe,size);
        pipe.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        if (!pipe) break;
        if (RGB){
            cv::Mat img(H, W, CV_8UC3, buffer.data());
            
            cv::imshow("Viewer", img);
        } else {
            cv::Mat img(H, W, CV_8UC1, buffer.data());

            cv::imshow("Viewer", img);
        }

        if (cv::waitKey(1) == 27) break; // ESC to exit
    }

    return 0;
}
