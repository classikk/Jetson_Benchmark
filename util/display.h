#ifndef DISPLAY_H
#define DISPLAY_H
//sudo apt install libopencv-dev
#include <opencv2/opencv.hpp>
//#include "util/image.h"

//void display(RGB888 image) {
//    // Wrap buffer into cv::Mat (no copy, just a header around your data)
//    cv::Mat img(image.height, image.width, CV_8UC3, image.start);
//
//    // OpenCV uses BGR order by default, so if your buffer is RGB, convert:
//    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
//
//    // Show the image
//    cv::imshow("Image Window", img);
//    cv::waitKey(0);
//}

void display() {
    int width = 640;   // your image width
    int height = 480;  // your image height

    // Suppose this is your raw RGB888 buffer in RAM
    unsigned char* buffer = new unsigned char[width * height * 3];

    // Fill buffer with something (for demo, set all pixels to red)
    for (int i = 0; i < width * height; i++) {
        buffer[3*i + 0] = 0;   // Blue
        buffer[3*i + 1] = 0;   // Green
        buffer[3*i + 2] = 255; // Red
    }

    // Wrap buffer into cv::Mat (no copy, just a header around your data)
    cv::Mat img(height, width, CV_8UC3, buffer);

    // OpenCV uses BGR order by default, so if your buffer is RGB, convert:
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

    // Show the image
    cv::imshow("Image Window", img);
    cv::waitKey(0);

    delete[] buffer;
}

#endif