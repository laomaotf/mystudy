#include <iostream>
#include <vector>

#include "stdio.h"
#include "stdlib.h"


#include "opencv2/opencv.hpp"


#include "dwt.h"

int main(int argc, char* argv[])
{
	std::string opencv_dir = getenv("OPENCV_PATH");
	cv::Mat image = cv::imread(opencv_dir + R"(\sources\samples\data\building.jpg)", 0);

	cv::Mat float_image;
	image.convertTo(float_image, CV_32F, 1.0, 0.0);

	int width = image.cols, height = image.rows;
	float* float_image_data = new float[width * height * 2];
	float* dwt_data = float_image_data + width * height;

	for (int y = 0; y < height; y++)
	{
		memcpy(
			float_image_data + y * width, float_image.data + y * float_image.step[0], width * sizeof(float)
		);
	}

	int depth = 3;
	int dwt_type = 3;
	dwt::forward(float_image_data, width, height, depth, dwt_type, dwt_data);

	cv::Mat dwt_image(height, width, CV_32FC1, dwt_data, width * sizeof(float));
	cv::Mat vis_dwt;
	cv::normalize(dwt_image, vis_dwt, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(vis_dwt, vis_dwt, 1.0, 0);
	cv::imshow("dwt", vis_dwt);


	memset(float_image_data, 0, sizeof(float) * width * height);
	dwt::backward(dwt_data, width, height, depth, dwt_type, float_image_data);


	cv::Mat idwt_image(height, width, CV_32FC1, float_image_data, width * sizeof(float));
	cv::Mat vis_idwt;
	cv::normalize(idwt_image, vis_idwt, 0, 255, cv::NORM_MINMAX);
	cv::convertScaleAbs(vis_idwt, vis_idwt, 1.0, 0);
	cv::imshow("idwt", vis_idwt);



	cv::waitKey(-1);

	delete[] float_image_data;
	return 0;
}
