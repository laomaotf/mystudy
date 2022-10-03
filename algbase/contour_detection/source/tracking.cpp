#include "contour_detection.h"

float EPS = 1e-6;

float TRACK_STEP_SCALE = 5;

std::vector<cv::Point> StartTrackAt(cv::Mat dx, cv::Mat dy, cv::Mat mag, cv::Mat dxdy,
	cv::Mat pixel_flag, cv::Point start_xy)
{
	int width = dx.cols, height = dx.rows;
	std::vector<cv::Point> contour;

	float Vx = 0, Vy = 0; //记录当前contour的延伸方向
	cv::Point pt = start_xy;
	while (1)
	{
		if (pt.x < 0 || pt.y < 0 || pt.x >= width || pt.y >= height) break;

		if (pixel_flag.at<uchar>(pt) > 0) break;

		contour.push_back(pt);

		pixel_flag.at<uchar>(pt) = 1;

		////////////////////////////////////////////////////////////////
		//计算contour方向变化量(轮廓方向逐渐和梯度方向垂直)
		// (dx,dy), rot90 -> (dy,dx), (dxdy * Vx, dxdy * Vy)]
		float ddx = dy.at<float>(pt) - dxdy.at<float>(pt) * Vx;
		float ddy = dx.at<float>(pt) - dxdy.at<float>(pt) * Vy;


		Vx += ddx;
		Vy += ddy;

		float m = std::sqrt(Vx * Vx + Vy * Vy);
		Vx /= (m + EPS);
		Vy /= (m + EPS);

		//更新位置
		pt.x += TRACK_STEP_SCALE * Vx;
		pt.y += TRACK_STEP_SCALE * Vy;

	}

	return contour;
}

std::vector<std::vector<cv::Point>> SortContours(std::vector<std::vector<cv::Point>>& contours)
{
	//argsort()
	std::vector<int> indices;
	for (int k = 0; k < contours.size(); k++)
	{
		indices.push_back(k);
	}

	std::sort(indices.begin(), indices.end(),
		[&contours](int k0, int k1) { return contours[k0].size() > contours[k1].size();  }
	);

	std::vector<std::vector<cv::Point>> sorted;
	for (auto pos : indices)
	{
		sorted.push_back(contours[pos]);
	}
	return sorted;
}

std::vector<std::vector<cv::Point>> GetContour(std::vector<cv::Mat>& grads, float seed )
{
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat dx = grads[0], dy = grads[1], dxdy = grads[2];
	int height = dx.rows, width = dx.cols;


	cv::Mat pixel_flag = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);


	cv::Mat mag = cv::Mat::zeros(cv::Size(width, height), CV_32FC1);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float gx = dx.at<float>(y, x), gy = dy.at<float>(y, x);
			mag.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
		}
	}
	float thresh_mag = cv::sum(mag)[0] / (width * height);
	thresh_mag = thresh_mag * seed;


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//closed pixel
			if (pixel_flag.at<uchar>(y, x)) continue;

			//weak edge
			if (mag.at<float>(y, x) < thresh_mag)
				continue;

			//remove smooth region
			cv::Rect roi(x - 2, y - 2, 5, 5);
			roi.x = std::max<int>(roi.x, 0);
			roi.y = std::max<int>(roi.y, 0);
			roi.width = std::min<int>(roi.width, width - roi.x);
			roi.height = std::min<int>(roi.height, height - roi.y);
			cv::Mat roi_data = mag(roi);
			double m0, m1;
			cv::minMaxIdx(roi_data, &m0, &m1);
			if (m1 > mag.at<float>(y, x)) continue; //flatten region

			//avoid duplicated scaning
			if (cv::sum(pixel_flag(roi))[0] > 0)
				continue;
	
			std::vector<cv::Point> contour = StartTrackAt(
				dx, dy, mag, dxdy, pixel_flag, cv::Point(x, y)
			);

			if (contour.empty()) continue;


			contours.push_back(contour);
		}
	}


	std::vector<std::vector<cv::Point>> sorted = SortContours(contours);
	return sorted;

}