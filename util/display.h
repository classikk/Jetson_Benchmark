#ifndef DISPLAY_H
#define DISPLAY_H
//sudo apt install libopencv-dev
#include <opencv2/opencv.hpp>
#include "image.h"

void display(RGB888 image) {


    // Wrap buffer into cv::Mat (no copy, just a header around your data)
    //cv::Mat img(image.height, image.width, CV_8UC3, image.start);

    // OpenCV uses BGR order by default, so if your buffer is RGB, convert:
    //cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    //cv::imshow("Image Window", img);
    
    cv::Mat imgRGB(image.height, image.width, CV_8UC3, image.start);

    // Display correctly by converting once into BGR
    cv::Mat imgBGR;
    cv::cvtColor(imgRGB, imgBGR, cv::COLOR_RGB2BGR);

    // Show the image
    cv::imshow("Image Window", imgBGR);
    cv::pollKey();
    //cv::waitKey(1);
}
#endif