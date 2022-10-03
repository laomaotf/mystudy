#pragma once
#include <vector>
#include "opencv2/opencv.hpp"

namespace watermark
{

	/*
	* 频域水印的思路是直接把图片set到频域中，即利用watermark图片像素值修改image的频域信息；
	* watermark图修改频域信息时采用随机编码，让像素信息分布到所有频率，有助于提高水印的稳定性
	*/
	struct INPUT
	{
		cv::Mat image;
		cv::Mat watermark;
		float alpha;
	};

	struct OUTPUT
	{
		cv::Mat image;
		cv::Mat watermak;
	};
	int Encode(INPUT& input, OUTPUT& output);
};
