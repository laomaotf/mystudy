#include "stdio.h"
#include "stdlib.h"
#include "contour_detection.h"
#include <algorithm>
#include <iterator>

int main(int argc, char* argv[])
{
	std::string opencv_dir = getenv("OPENCV_PATH");
	cv::Mat image = cv::imread(opencv_dir + R"(\sources\samples\data\building.jpg)",0);

	//image = 255 - image; 

	std::vector<cv::Mat> grads = GetGradient(image,true);

	std::vector<std::string> grads_name = { "abs(dx)", "abs(dy)", "abs(dxdy)" };
	for (int k = 0; k < 3; k++)
	{
		cv::Mat vis;

		vis = cv::abs(grads[k]);
		cv::normalize(vis, vis, 255, 0, cv::NORM_MINMAX);
		cv::convertScaleAbs(vis, vis, 1.0, 0.0);
		cv::imshow(grads_name[k], vis);
	}
	cv::waitKey(200);


	std::vector< std::vector<cv::Point>> contours = GetContour(grads,0.75);

	

	for (auto& contour : contours)
	{
		if (contour.size() < 30) continue;
		cv::Mat vis;
		cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
		for (auto& pt : contour)
			cv::circle(vis, pt, 1, CV_RGB(255, 0, 0), 1);

		cv::imshow("contour", vis);
		cv::waitKey(200);
	}

	return 0;
}