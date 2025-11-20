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
    
    cv::Mat imgRGB(image.height, image.width, CV_8UC3, image.start);

    // Display correctly by converting once into BGR
    cv::Mat imgBGR;
    cv::cvtColor(imgRGB, imgBGR, cv::COLOR_RGB2BGR);

    // Show the image
    cv::imshow("RGB888", imgBGR);
    cv::pollKey();
    //cv::waitKey(1);
}

void display(RG10 image) {

    cv::Mat raw(image.height, image.width, CV_16UC1, image.start);

    // Display correctly by converting once into BGR
    cv::Mat bgr;
    cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);

    cv::Mat enhanced;
    bgr.convertTo(enhanced, -1, 3.5, 100);

    //cv::Mat bgr8;
    //bgr.convertTo(bgr8, CV_8U, 255.0 / 1023.0);
//
    //cv::Mat lab;
    //cv::cvtColor(bgr8, lab, cv::COLOR_BGR2Lab);
    //std::vector<cv::Mat> labChannels;
    //cv::split(lab, labChannels);
//
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
    //clahe->apply(labChannels[0], labChannels[0]);
//
    //cv::merge(labChannels, lab);
    //cv::Mat enhanced;
    //cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);

    //cv::Mat bgr8;
    //bgr.convertTo(bgr8, CV_8U, 255.0/1023.0);
    // Show the image
    cv::imshow("RG10", enhanced);
    cv::pollKey();
    //cv::waitKey(1);
}
#endif