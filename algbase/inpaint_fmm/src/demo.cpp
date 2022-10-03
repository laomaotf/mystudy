#include "stdio.h"
#include "inpaint.h"

std::vector<cv::Mat> PrepareDefect(cv::Mat img)
{
	int width = img.cols, height = img.rows;
	cv::Point p0(rand() % (width/2 - 50) + 20, rand() % (height/2 - 50) + 20);
	cv::Point p1(rand() % (width/2 - 50) + width/2, rand() % (height/2 - 50) + height/2);
	cv::Point p3(width/2, height/2);

	cv::Mat defect_image = img.clone();
	cv::Mat defect_mask = cv::Mat::zeros(img.size(), CV_8UC1);
	cv::line(defect_image, p0, p1, CV_RGB(255, 255, 255), 1);

	cv::circle(defect_image, p3, rand() % 100 + 10, CV_RGB(255, 255, 255));

	cv::circle(defect_image, p3, rand() % 100 + 10, CV_RGB(255, 255, 255),3);
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (img.at<uchar>(y, x) != defect_image.at<uchar>(y, x))
				defect_mask.at<uchar>(y, x) = 255;
		}
	}
	
	cv::dilate(defect_mask, defect_mask, cv::Mat(), cv::Point(-1,-1),1); //扩大一个像素，避免处理单像素

	return { defect_image, defect_mask };
}

int main(int nargc, char* argv[])
{
	std::string path = getenv("OpenCV_DIR");
	path += "\\..\\sources\\samples\\data\\lena.jpg";
	cv::Mat raw = cv::imread(path, 0);
	std::vector<cv::Mat> defect = PrepareDefect(raw);
	cv::Mat image = defect[0], mask = defect[1];
	cv::imshow("image", image);
	cv::imshow("mask", mask);
	cv::Mat fixed_image = DoInpaint(image, mask);

	cv::imshow("fixed", fixed_image);
	cv::waitKey(-1);
	return 0;
}