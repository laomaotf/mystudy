#include <iostream>
#include <vector>
#include "stroke_width_transform.h"

int main(int argc, char* argv[])
{
	std::string image_path = getenv("OPENCV_DIR");
	image_path += "\\..\\sources\\samples\\data\\sudoku.png";


	cv::Mat image = cv::imread(image_path, 0);
	cv::Mat image_reversed = 255 - image;

	std::vector<cv::Rect> texts;
	SWTTextDetection(image_reversed, texts);

	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	for (auto& r : texts)
	{
		cv::rectangle(image, r, CV_RGB(255, 0, 0), 1);
	}
	cv::imshow("vis", image);
	cv::waitKey(-1);

	return 0;
}