#ifndef DISPLAY_H
#define DISPLAY_H
//sudo apt install libopencv-dev
#include <opencv2/opencv.hpp>
#include "image.h"

using namespace cv;
void display(RGB888 image) {
    // Wrap buffer into cv::Mat (no copy, just a header around your data)
    //cv::Mat img(image.height, image.width, CV_8UC3, image.start);

    // OpenCV uses BGR order by default, so if your buffer is RGB, convert:
    //cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    //cv::imshow("Image Window", img);
    
    cv::Mat imgRGB(image.info.height, image.info.width, CV_8UC3, image.start);

    // Display correctly by converting once into BGR
    //cv::Mat imgBGR;
    //cv::cvtColor(imgRGB, imgBGR, cv::COLOR_RGB2BGR);

    // Show the image
    //cv::imshow("RGB888", imgBGR);
    cv::imshow("RGB888", imgRGB);
    cv::pollKey();
}

void display(RG10 image) {

    cv::Mat raw(image.info.height, image.info.width, CV_16UC1, image.start);

    // Display correctly by converting once into BGR
    cv::Mat bgr;
    cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);

    //cv::Mat enhanced;
    //bgr.convertTo(enhanced, -1, 3.5, 100);


    // Show the image
    cv::imshow("RG10", bgr);
    cv::pollKey();
}

void display(BW8 image) {
    cv::Mat imgBW8(image.info.height, image.info.width, CV_8UC1, image.start);
    cv::imshow("BW8", imgBW8);
    cv::pollKey();
}
#endif