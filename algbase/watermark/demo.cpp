#include "stdio.h"
#include "stdlib.h"
#include "watermark.h"



int main(int argc, char* argv[])
{

	std::string opencv_dir = getenv("OPENCV_PATH"); //读取环境变量

	cv::Mat image = cv::imread(opencv_dir + R"(\sources\samples\data\smarties.png)", 1);
	cv::Mat wm = cv::imread(opencv_dir + R"(\sources\samples\data\LinuxLogo.jpg)", 0);

	watermark::INPUT input;
	watermark::OUTPUT output;

	input.image = image;
	input.watermark = wm;
	input.alpha = 50.0;
	watermark::Encode(input, output);

	cv::imshow("image", image);
	cv::imshow("watermark", wm);

	cv::imshow("image_with_watermark", output.image);
	cv::imshow("rebuild watermark", output.watermak);

	cv::waitKey(-1);
	return 0;
} 