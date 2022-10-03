#pragma once
#include "opencv2/opencv.hpp"

std::vector<cv::Mat> GetGradient(cv::Mat image,bool dark_edge = false);
std::vector<std::vector<cv::Point>> GetContour(std::vector<cv::Mat>& grads, float seed = 0.5);
