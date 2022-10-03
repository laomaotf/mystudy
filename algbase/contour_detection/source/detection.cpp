#include "contour_detection.h"

int SOBEL_KERNEL_SIZE = 3;
int DOG_KERNEL_G1 = 7;
int DOG_KERNEL_G2 = 5;


float KERNEL_DXDY[] =
{
	-1.5631f, -2.3419f, -1.5609f, -2.3419f, -1.5631f,
	-2.3419f, 0.0f,        4.6838f, 0.0f,       -2.3419f,
	-1.5609f, 4.6838f,   12.4938f,4.6838f,  -1.5609f,
	-2.3149f, 0.0f,        4.6838f, 0.0f,       -2.3419f,
	-1.5631f, -2.3419f,  -1.5609f,-2.3419f, -1.5631f,
};

float KERNEL_DXNEG[] =
{
	0.4253f, 0.4617f, 0.0f, -0.4617f, -0.4253f,
	0.6900f, 0.9895f, 0.0f, -0.9895f, -0.6900f,
	0.1521f, 0.4880f, 0.0f, -0.4880f, -0.1521f,
	0.6900f, 0.9895f, 0.0f, -0.9895f, -0.6900f,
	0.4253f, 0.4617f, 0.0f, -0.4617f, -0.4253f,
};

float KERNEL_DX[] =
{
	-0.4253f, -0.4617f, 0.0f, 0.4617f, 0.4253f,
	-0.6900f, -0.9895f, 0.0f, 0.9895f, 0.6900f,
	-0.1521f, -0.4880f, 0.0f, 0.4880f, 0.1521f,
	-0.6900f, -0.9895f, 0.0f, 0.9895f, 0.6900f,
	-0.4253f, -0.4617f, 0.0f, 0.4617f, 0.4253f,
};

float KERNEL_DY[] =
{
	-0.4253f, -0.6900f, -0.1521f, -0.6900f, -0.4253f,
	-0.4617f, -0.9896f, -0.4880f, -0.9895f, -0.4617f,
	0.0f,        0.0f,       0.0f,       0.0f,		0.0f,
	0.4617f, 0.9896f, 0.4880f, 0.9895f, 0.4617f,
	0.4253f, 0.6900f, 0.1521f, 0.6900f, 0.4253f,
};


float KERNEL_DYNEG[] =
{
	0.4253f, 0.6900f, 0.1521f, 0.6900f, 0.4253f,
	0.4617f, 0.9896f, 0.4880f, 0.9895f, 0.4617f,
	0.0f,        0.0f,       0.0f,       0.0f,		0.0f,
	-0.4617f, -0.9896f, -0.4880f, -0.9895f, -0.4617f,
	-0.4253f, -0.6900f, -0.1521f, -0.6900f, -0.4253f,
};


cv::Mat RunFilter(cv::Mat image, float* data, cv::Size size)
{
	cv::Mat kernel(size.width, size.height, CV_32F, data, sizeof(float) * size.width);
	cv::Mat grad;
	cv::filter2D(image, grad, CV_32F, kernel);
	float norm = std::abs<float>(cv::sum(kernel)[0]);
	return grad / (norm + 1e-5); //规范梯度取值范围
}


std::vector<cv::Mat> GetGradient(cv::Mat image, bool dark_edge)
{
	int width = image.cols, height = image.rows;
	cv::Mat dx, dy, dxdy;

	if (dark_edge)
	{//适合阴文
		dx = RunFilter(image, KERNEL_DXNEG, cv::Size(5, 5));
		dy = RunFilter(image, KERNEL_DY, cv::Size(5, 5));
	}
	else
	{//适合阳文
		dx = RunFilter(image, KERNEL_DX, cv::Size(5, 5));
		dy = RunFilter(image, KERNEL_DYNEG, cv::Size(5, 5));
	}


	dxdy = RunFilter(image, KERNEL_DXDY, cv::Size(5, 5));

	std::vector<cv::Mat> outputs = { dx,dy,dxdy };
	return outputs;
}